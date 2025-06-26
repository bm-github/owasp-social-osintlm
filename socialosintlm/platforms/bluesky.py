import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, cast
from urllib.parse import quote_plus, urlparse

from atproto import Client, exceptions as atproto_exceptions
# from atproto_client.models.app.bsky.feed.post import Record as PostRecordType
# from atproto_client.models.app.bsky.richtext.facet import Main as FacetType

from ..cache import CACHE_EXPIRY_HOURS, MAX_CACHE_ITEMS, CacheManager
from ..exceptions import (AccessForbiddenError, RateLimitExceededError,
                         UserNotFoundError)
from ..llm import LLMAnalyzer
from ..utils import SUPPORTED_IMAGE_EXTENSIONS, download_media, get_sort_key

logger = logging.getLogger("SocialOSINTLM.platforms.bluesky")

# Default used if no count is specified in the fetch plan
DEFAULT_FETCH_LIMIT = 50

def fetch_data(
    client: Client,
    username: str,
    cache: CacheManager,
    llm: LLMAnalyzer,
    force_refresh: bool = False,
    fetch_limit: int = DEFAULT_FETCH_LIMIT,
) -> Optional[Dict[str, Any]]:
    """Fetches posts and user profile for a Bluesky user."""
    
    cached_data = cache.load("bluesky", username)

    if cache.is_offline:
        return cached_data or {"timestamp": datetime.now(timezone.utc).isoformat(), "profile_info": {}, "posts": [], "media_analysis": [], "media_paths": [], "stats": {}}

    if not force_refresh and cached_data and (datetime.now(timezone.utc) - get_sort_key(cached_data, "timestamp")) < timedelta(hours=CACHE_EXPIRY_HOURS):
        if len(cached_data.get("posts", [])) >= fetch_limit:
            return cached_data

    logger.info(f"Fetching Bluesky data for {username} (Force Refresh: {force_refresh}, Limit: {fetch_limit})")
    
    existing_posts = cached_data.get("posts", []) if not force_refresh and cached_data else []
    # Only check for new posts if we aren't trying to "load more"
    use_incremental_fetch = not force_refresh and fetch_limit <= len(existing_posts)
    latest_post_datetime = get_sort_key(existing_posts[0], "created_at") if use_incremental_fetch and existing_posts else None

    profile_info = cached_data.get("profile_info") if not force_refresh and cached_data else None
    existing_media_analysis = cached_data.get("media_analysis", []) if not force_refresh and cached_data else []
    existing_media_paths = cached_data.get("media_paths", []) if not force_refresh and cached_data else []

    try:
        if not profile_info or force_refresh:
            profile = client.get_profile(actor=username)
            labels_list = [{"value": lbl.val, "timestamp": lbl.cts} for lbl in profile.labels] if profile.labels else []
            profile_info = {
                "did": profile.did, "handle": profile.handle, "display_name": profile.display_name,
                "description": profile.description, "avatar": profile.avatar, "banner": profile.banner,
                "followers_count": profile.followers_count, "follows_count": profile.follows_count,
                "posts_count": profile.posts_count, "labels": labels_list
            }

        new_posts_data = []
        newly_added_media_analysis = []
        newly_added_media_paths = set()
        did_to_handle_cache: Dict[str, str] = {profile_info["did"]: profile_info["handle"]}
        
        cursor = None
        total_fetched = 0
        max_fetches = fetch_limit

        auth_details = {"access_jwt": getattr(client._session, 'access_jwt', None)}

        while total_fetched < max_fetches:
            response = client.get_author_feed(actor=username, cursor=cursor, limit=min(max_fetches - total_fetched, 100))
            if not response or not response.feed: break
            
            for feed_item in response.feed:
                post = feed_item.post
                record = cast(Any, post.record)
                if not record: continue # Skip if post has no record data
                
                created_at_dt = get_sort_key({"created_at": getattr(record, "created_at", None)}, "created_at")

                # If doing incremental, stop when we see a post we already have.
                if latest_post_datetime and created_at_dt <= latest_post_datetime:
                    cursor = "STOP" # Signal to stop
                    break
                
                media_items, did_to_handle_cache = _process_post_media(record, post, cache, llm, username, auth_details, newly_added_media_paths, newly_added_media_analysis)
                
                reply_info = _get_reply_info(record, client, did_to_handle_cache)
                embed_info = _get_embed_info(record, client, did_to_handle_cache)
                mentions = _get_mentions(record, did_to_handle_cache)

                post_data = {
                    "uri": post.uri, "cid": post.cid, "author_did": post.author.did,
                    "text": getattr(record, "text", ""), "created_at": created_at_dt.isoformat(),
                    "langs": getattr(record, "langs", []), "likes": post.like_count, 
                    "reposts": post.repost_count, "reply_count": post.reply_count,
                    "media": media_items, "mentions": mentions, **reply_info, **embed_info
                }
                new_posts_data.append(post_data)
                total_fetched += 1
            
            if cursor == "STOP" or not response.cursor: break
            cursor = response.cursor

        combined = new_posts_data + existing_posts
        final_posts = sorted(list({p['uri']: p for p in combined}.values()), key=lambda x: get_sort_key(x, "created_at"), reverse=True)[:max(fetch_limit, MAX_CACHE_ITEMS)]
        final_media_analysis = sorted(list(set(newly_added_media_analysis + existing_media_analysis)))
        final_media_paths = sorted(list(newly_added_media_paths.union(existing_media_paths)))

        stats = {
            "total_posts_cached": len(final_posts),
            "posts_with_media": len([p for p in final_posts if p.get("media")]),
            "reply_posts_cached": len([p for p in final_posts if p.get("reply_parent_uri")]),
            "avg_likes": round(sum(p.get("likes", 0) for p in final_posts) / max(1, len(final_posts)), 2)
        }

        final_data = {"profile_info": profile_info, "posts": final_posts, "stats": stats, "media_analysis": final_media_analysis, "media_paths": final_media_paths[:MAX_CACHE_ITEMS*2]}
        cache.save("bluesky", username, final_data)
        return final_data

    except atproto_exceptions.AtProtocolError as e:
        # Correctly check for rate limit using isinstance
        if isinstance(e, atproto_exceptions.RateLimitExceeded):
            raise RateLimitExceededError("Bluesky API rate limit exceeded.") from e
            
        err_str = str(e).lower()
        if "profile not found" in err_str or "could not resolve handle" in err_str:
            raise UserNotFoundError(f"Bluesky user {username} not found.") from e
        if "blocked by actor" in err_str:
            raise AccessForbiddenError(f"Access to Bluesky user {username} is blocked.") from e
            
        logger.error(f"Bluesky API error for {username}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching Bluesky data for {username}: {e}", exc_info=True)
        return None

# Helper functions for Bluesky data processing (unchanged)
def _get_reply_info(record: Any, client: Client, did_cache: Dict[str, str]) -> Dict[str, Any]:
    reply_ref = getattr(record, "reply", None)
    if not reply_ref: return {}
    
    parent_uri = getattr(reply_ref.parent, "uri", None)
    root_uri = getattr(reply_ref.root, "uri", None)
    parent_author_handle = None
    if parent_uri:
        parent_did = urlparse(parent_uri).netloc
        parent_author_handle = _resolve_did(parent_did, client, did_cache)

    return {"reply_parent_uri": parent_uri, "reply_root_uri": root_uri, "reply_parent_author_handle": parent_author_handle}

def _get_embed_info(record: Any, client: Client, did_cache: Dict[str, str]) -> Dict[str, Any]:
    embed = getattr(record, "embed", None)
    if not embed: return {}
    
    embed_type = getattr(embed, "$type", None)
    embedded_post_author_handle = None
    
    record_field = getattr(embed, 'record', None)
    if "embed.record" in str(embed_type) and record_field and hasattr(record_field, 'author'):
        author_did = getattr(record_field.author, 'did', None)
        if author_did:
            embedded_post_author_handle = _resolve_did(author_did, client, did_cache)

    return {"embed_type": embed_type, "embedded_post_author_handle": embedded_post_author_handle}

def _get_mentions(record: Any, did_cache: Dict[str, str]) -> List[Dict[str, str]]:
    mentions = []
    # Add a check for None before iterating
    facets = getattr(record, "facets", None)
    if facets:
        for facet in facets:
            features = getattr(facet, "features", None)
            if features:
                for feature in features:
                    if getattr(feature, '$type', '') == 'app.bsky.richtext.facet#mention':
                        did = getattr(feature, 'did', None)
                        if did:
                            handle = did_cache.get(did, did) # Fallback to DID
                            mentions.append({"did": did, "handle": handle})
    return mentions

def _process_post_media(record: Any, post: Any, cache: CacheManager, llm: LLMAnalyzer, username: str, auth: Dict, paths: set, analyses: list) -> tuple[list, dict]:
    media_items = []
    embed = getattr(record, "embed", None)
    images_to_process = []
    if embed:
        if hasattr(embed, "images"): images_to_process.extend(embed.images)
        record_media = getattr(embed, 'media', None)
        if record_media and hasattr(record_media, 'images'):
            images_to_process.extend(record_media.images)

    for image_info in images_to_process:
        img_blob = getattr(image_info, "image", None)
        if img_blob:
            cid = getattr(img_blob, "cid", None)
            if cid:
                mime_type = getattr(img_blob, "mime_type", "image/jpeg").split('/')[-1]
                cdn_url = f"https://cdn.bsky.app/img/feed_fullsize/plain/{post.author.did}/{quote_plus(str(cid))}@{mime_type}"
                media_path = download_media(cache.base_dir, cdn_url, cache.is_offline, "bluesky", auth)
                if media_path:
                    analysis = llm.analyze_image(media_path, f"Bluesky user {username}'s post") if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else None
                    media_items.append({"type": "image", "analysis": analysis, "url": cdn_url, "alt_text": image_info.alt, "local_path": str(media_path)})
                    if analysis: analyses.append(analysis)
                    paths.add(str(media_path))
    return media_items, {}

def _resolve_did(did: str, client: Client, cache: Dict[str, str]) -> Optional[str]:
    if did in cache:
        return cache[did]
    try:
        if hasattr(client, '_session') and client._session:
            profile = client.get_profile(actor=did)
            if profile and profile.handle:
                cache[did] = profile.handle
                return profile.handle
    except atproto_exceptions.AtProtocolError:
        pass
    return did