import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from mastodon import (Mastodon, MastodonError, MastodonNotFoundError,
                    MastodonRatelimitError)

from ..cache import CACHE_EXPIRY_HOURS, MAX_CACHE_ITEMS, CacheManager
from ..exceptions import (AccessForbiddenError, RateLimitExceededError,
                         UserNotFoundError)
from ..llm import LLMAnalyzer
from ..utils import SUPPORTED_IMAGE_EXTENSIONS, download_media, get_sort_key

logger = logging.getLogger("SocialOSINTLM.platforms.mastodon")

# Default used if no count is specified in the fetch plan
DEFAULT_FETCH_LIMIT = 40

def fetch_data(
    clients: Dict[str, Mastodon],
    default_client: Optional[Mastodon],
    username: str, # user@instance.domain
    cache: CacheManager,
    llm: LLMAnalyzer,
    force_refresh: bool = False,
    fetch_limit: int = DEFAULT_FETCH_LIMIT
) -> Optional[Dict[str, Any]]:
    """Fetches statuses and user info for a Mastodon user."""
    
    if "@" not in username or len(username.split('@', 1)) != 2:
        raise ValueError(f"Invalid Mastodon username format: '{username}'. Must be 'user@instance.domain'.")

    cached_data = cache.load("mastodon", username)

    if cache.is_offline:
        return cached_data or {"timestamp": datetime.now(timezone.utc).isoformat(), "user_info": {}, "posts": [], "media_analysis": [], "media_paths": [], "stats": {}}

    if not force_refresh and cached_data and (datetime.now(timezone.utc) - get_sort_key(cached_data, "timestamp")) < timedelta(hours=CACHE_EXPIRY_HOURS):
        if len(cached_data.get("posts", [])) >= fetch_limit:
            return cached_data

    logger.info(f"Fetching Mastodon data for {username} (Force Refresh: {force_refresh}, Limit: {fetch_limit})")
    
    instance_domain = username.split('@')[1]
    client_to_use = clients.get(f"https://{instance_domain}") or default_client
    if not client_to_use:
        raise RuntimeError(f"No suitable Mastodon client found for instance {instance_domain} or for default lookup.")

    existing_posts = cached_data.get("posts", []) if not force_refresh and cached_data else []
    since_id = existing_posts[0].get("id") if existing_posts and fetch_limit <= len(existing_posts) else None
    user_info = cached_data.get("user_info") if not force_refresh and cached_data else None
    existing_media_analysis = cached_data.get("media_analysis", []) if not force_refresh and cached_data else []
    existing_media_paths = cached_data.get("media_paths", []) if not force_refresh and cached_data else []

    try:
        if not user_info or force_refresh:
            account = client_to_use.account_lookup(acct=username)
            user_info = {
                "id": str(account["id"]), "username": account["username"], "acct": account["acct"],
                "display_name": account["display_name"], "url": account["url"],
                "note_text": BeautifulSoup(account.get("note",""), "html.parser").get_text(separator=" ", strip=True),
                "followers_count": account["followers_count"], "following_count": account["following_count"],
                "statuses_count": account["statuses_count"], "locked": account.get("locked", False),
                "bot": account.get("bot", False), "created_at": account["created_at"].isoformat()
            }
        
        user_id = user_info["id"]
        # The Mastodon API has a hard limit of 40 per request.
        # To fetch more, pagination would be required. For now, we respect the API limit.
        api_limit = min(fetch_limit, 40) 
        new_statuses = client_to_use.account_statuses(id=user_id, limit=api_limit, since_id=since_id if not force_refresh else None)
        
        new_posts_data = []
        newly_added_media_analysis = []
        newly_added_media_paths = set()

        for status in new_statuses:
            cleaned_text = BeautifulSoup(status["content"], "html.parser").get_text(separator=" ", strip=True)
            media_items = []
            for att in status.get("media_attachments", []):
                media_path = download_media(cache.base_dir, att["url"], cache.is_offline, "mastodon")
                if media_path:
                    analysis = llm.analyze_image(media_path, f"Mastodon user {username}'s post") if att["type"] == "image" else None
                    media_items.append({"id": str(att["id"]), "type": att["type"], "analysis": analysis, "url": att["url"], "description": att.get("description"), "local_path": str(media_path)})
                    if analysis: newly_added_media_analysis.append(analysis)
                    newly_added_media_paths.add(str(media_path))
            
            reblog_info = status.get("reblog")
            poll_info = status.get("poll")

            post_data = {
                "id": str(status["id"]), "created_at": status["created_at"].isoformat(), "url": status["url"],
                "text_cleaned": cleaned_text, "visibility": status.get("visibility"), "sensitive": status.get("sensitive"),
                "spoiler_text": status.get("spoiler_text", ""), "language": status.get("language"),
                "reblogs_count": status.get("reblogs_count", 0), "favourites_count": status.get("favourites_count", 0),
                "is_reblog": reblog_info is not None,
                "reblog_original_author_acct": reblog_info['account']['acct'] if reblog_info else None,
                "reblog_original_url": reblog_info['url'] if reblog_info else None,
                "tags": [{"name": t["name"], "url": t["url"]} for t in status.get("tags", [])],
                "mentions": [{"acct": m["acct"], "url": m["url"]} for m in status.get("mentions", [])],
                "poll": {"options": poll_info["options"], "votes_count": poll_info.get("votes_count")} if poll_info else None,
                "media": media_items
            }
            new_posts_data.append(post_data)

        combined = new_posts_data + existing_posts
        final_posts = sorted(list({p['id']: p for p in combined}.values()), key=lambda x: get_sort_key(x, "created_at"), reverse=True)[:max(fetch_limit, MAX_CACHE_ITEMS)]
        final_media_analysis = sorted(list(set(newly_added_media_analysis + existing_media_analysis)))
        final_media_paths = sorted(list(newly_added_media_paths.union(existing_media_paths)))

        stats = {
            "total_posts_cached": len(final_posts),
            "total_original_posts_cached": len([p for p in final_posts if not p.get("is_reblog")]),
            "total_reblogs_cached": len([p for p in final_posts if p.get("is_reblog")]),
            "posts_with_media": len([p for p in final_posts if p.get("media")]),
        }

        final_data = {"user_info": user_info, "posts": final_posts, "stats": stats, "media_analysis": final_media_analysis, "media_paths": final_media_paths[:MAX_CACHE_ITEMS*2]}
        cache.save("mastodon", username, final_data)
        return final_data

    except MastodonRatelimitError:
        raise RateLimitExceededError("Mastodon API rate limit exceeded.")
    except MastodonNotFoundError:
        raise UserNotFoundError(f"Mastodon user {username} not found.")
    except MastodonError as e:
        err_str = str(e).lower()
        if "unauthorized" in err_str or "forbidden" in err_str or "locked" in err_str:
            raise AccessForbiddenError(f"Access to Mastodon user {username} is not authorized (locked account?).") from e
        logger.error(f"Mastodon API error for {username}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching Mastodon data for {username}: {e}", exc_info=True)
        return None