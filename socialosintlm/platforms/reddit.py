import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import praw
import prawcore

from ..cache import CACHE_EXPIRY_HOURS, MAX_CACHE_ITEMS, CacheManager
from ..exceptions import (AccessForbiddenError, RateLimitExceededError,
                         UserNotFoundError)
from ..llm import LLMAnalyzer
from ..utils import SUPPORTED_IMAGE_EXTENSIONS, download_media, get_sort_key

logger = logging.getLogger("SocialOSINTLM.platforms.reddit")

# Default used if no count is specified in the fetch plan
DEFAULT_FETCH_LIMIT = 50

def fetch_data(
    client: praw.Reddit,
    username: str,
    cache: CacheManager,
    llm: LLMAnalyzer,
    force_refresh: bool = False,
    fetch_limit: int = DEFAULT_FETCH_LIMIT,
) -> Optional[Dict[str, Any]]:
    """Fetches submissions, comments, and user profile for a Reddit user."""
    
    cached_data = cache.load("reddit", username)

    if cache.is_offline:
        if cached_data:
            return cached_data
        else:
            return {"timestamp": datetime.now(timezone.utc).isoformat(), "user_profile": {}, "submissions": [], "comments": [], "media_analysis": [], "media_paths": [], "stats": {}}

    if not force_refresh and cached_data and (datetime.now(timezone.utc) - get_sort_key(cached_data, "timestamp")) < timedelta(hours=CACHE_EXPIRY_HOURS):
        # If the user has enough items cached, return the cache
        if len(cached_data.get("submissions", [])) >= fetch_limit and len(cached_data.get("comments", [])) >= fetch_limit:
            return cached_data

    logger.info(f"Fetching Reddit data for u/{username} (Force Refresh: {force_refresh}, Limit: {fetch_limit})")
    
    existing_submissions = cached_data.get("submissions", []) if not force_refresh and cached_data else []
    existing_comments = cached_data.get("comments", []) if not force_refresh and cached_data else []
    
    # Only use 'before' for incremental if we're not trying to fetch more than we already have
    use_incremental_subs = not force_refresh and fetch_limit <= len(existing_submissions)
    use_incremental_comments = not force_refresh and fetch_limit <= len(existing_comments)

    latest_submission_fullname = existing_submissions[0].get("fullname") if use_incremental_subs and existing_submissions else None
    latest_comment_fullname = existing_comments[0].get("fullname") if use_incremental_comments and existing_comments else None
    
    user_profile = cached_data.get("user_profile") if not force_refresh and cached_data else None
    existing_media_analysis = cached_data.get("media_analysis", []) if not force_refresh and cached_data else []
    existing_media_paths = cached_data.get("media_paths", []) if not force_refresh and cached_data else []

    try:
        redditor = client.redditor(username)
        if not user_profile or force_refresh:
            user_profile = {
                "id": redditor.id, "name": redditor.name,
                "created_utc": datetime.fromtimestamp(redditor.created_utc, tz=timezone.utc).isoformat(),
                "link_karma": redditor.link_karma, "comment_karma": redditor.comment_karma,
                "icon_img": redditor.icon_img, "is_suspended": getattr(redditor, "is_suspended", False)
            }
        
        newly_added_media_analysis = []
        newly_added_media_paths = set()

        # Fetch submissions
        new_submissions = []
        params_subs = {"limit": fetch_limit}
        if latest_submission_fullname:
            params_subs["before"] = latest_submission_fullname
        for s in redditor.submissions.new(**params_subs):
            media_items = []
            # Direct media link
            if any(s.url.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS + [".mp4", ".webm"]):
                media_path = download_media(cache.base_dir, s.url, cache.is_offline, "reddit")
                if media_path:
                    analysis = llm.analyze_image(media_path, f"Reddit user u/{username}'s post in r/{s.subreddit.display_name}") if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else None
                    media_items.append({"type": "image" if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else "video", "analysis": analysis, "url": s.url, "local_path": str(media_path)})
                    if analysis: newly_added_media_analysis.append(analysis)
                    newly_added_media_paths.add(str(media_path))
            # Gallery
            elif getattr(s, 'is_gallery', False) and getattr(s, 'media_metadata', None):
                for media_id, media_item in s.media_metadata.items():
                    if 's' in media_item and 'u' in media_item['s']:
                        url = media_item['s']['u']
                        media_path = download_media(cache.base_dir, url, cache.is_offline, "reddit")
                        if media_path:
                            analysis = llm.analyze_image(media_path, f"Reddit gallery post by u/{username}") if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else None
                            media_items.append({"type": "gallery_image", "analysis": analysis, "url": url, "local_path": str(media_path)})
                            if analysis: newly_added_media_analysis.append(analysis)
                            newly_added_media_paths.add(str(media_path))

            new_submissions.append({
                "id": s.id, "fullname": s.fullname, "title": s.title,
                "text": s.selftext, "score": s.score, "upvote_ratio": s.upvote_ratio,
                "subreddit": s.subreddit.display_name, "permalink": f"https://www.reddit.com{s.permalink}",
                "created_utc": datetime.fromtimestamp(s.created_utc, tz=timezone.utc).isoformat(),
                "link_url": s.url if not s.is_self else None,
                "is_self": s.is_self, "over_18": s.over_18, "spoiler": s.spoiler,
                "num_comments": s.num_comments, "media": media_items
            })
        
        # Fetch comments
        new_comments = []
        params_comments = {"limit": fetch_limit}
        if latest_comment_fullname:
            params_comments["before"] = latest_comment_fullname
        for c in redditor.comments.new(**params_comments):
            parent_author = c.submission.author.name if hasattr(c.submission, 'author') and c.submission.author else None
            new_comments.append({
                "id": c.id, "fullname": c.fullname, "text": c.body, "score": c.score,
                "subreddit": c.subreddit.display_name, "permalink": f"https://www.reddit.com{c.permalink}",
                "created_utc": datetime.fromtimestamp(c.created_utc, tz=timezone.utc).isoformat(),
                "is_submitter": c.is_submitter, "parent_submission_author": parent_author
            })
        
        # Combine and save
        combined_submissions = new_submissions + existing_submissions
        final_submissions = list({s['id']: s for s in combined_submissions}.values())
        final_submissions.sort(key=lambda x: get_sort_key(x, "created_utc"), reverse=True)
        
        combined_comments = new_comments + existing_comments
        final_comments = list({c['id']: c for c in combined_comments}.values())
        final_comments.sort(key=lambda x: get_sort_key(x, "created_utc"), reverse=True)

        final_media_analysis = sorted(list(set(newly_added_media_analysis + existing_media_analysis)))
        final_media_paths = sorted(list(newly_added_media_paths.union(existing_media_paths)))

        stats = {
            "total_submissions_cached": len(final_submissions),
            "total_comments_cached": len(final_comments),
            "submissions_with_media": len([s for s in final_submissions if s.get("media")]),
            "total_media_items_processed": len(final_media_paths),
            "avg_submission_score": round(sum(s.get("score", 0) for s in final_submissions) / max(1, len(final_submissions)), 2),
            "avg_comment_score": round(sum(c.get("score", 0) for c in final_comments) / max(1, len(final_comments)), 2),
            "avg_submission_upvote_ratio": round(sum(s.get("upvote_ratio", 0.0) or 0.0 for s in final_submissions) / max(1, len(final_submissions)), 3)
        }

        final_data = {
            "user_profile": user_profile, 
            "submissions": final_submissions[:MAX_CACHE_ITEMS], 
            "comments": final_comments[:MAX_CACHE_ITEMS],
            "media_analysis": final_media_analysis, 
            "media_paths": final_media_paths[:MAX_CACHE_ITEMS*2], 
            "stats": stats
        }
        cache.save("reddit", username, final_data)
        return final_data

    except prawcore.exceptions.NotFound:
        raise UserNotFoundError(f"Reddit user u/{username} not found.")
    except prawcore.exceptions.Forbidden:
        raise AccessForbiddenError(f"Access to Reddit user u/{username} is forbidden.")
    except prawcore.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
            raise RateLimitExceededError("Reddit API rate limit exceeded.") from e
        logger.error(f"Reddit request failed for u/{username}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching Reddit data for u/{username}: {e}", exc_info=True)
        return None