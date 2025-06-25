import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import tweepy

from ..cache import CACHE_EXPIRY_HOURS, MAX_CACHE_ITEMS, CacheManager
from ..exceptions import (AccessForbiddenError, RateLimitExceededError,
                         UserNotFoundError)
from ..llm import LLMAnalyzer
from ..utils import SUPPORTED_IMAGE_EXTENSIONS, download_media, get_sort_key

logger = logging.getLogger("SocialOSINTLM.platforms.twitter")

INITIAL_FETCH_LIMIT = 50
INCREMENTAL_FETCH_LIMIT = 50

def fetch_data(
    client: tweepy.Client,
    username: str,
    cache: CacheManager,
    llm: LLMAnalyzer,
    force_refresh: bool = False,
) -> Optional[Dict[str, Any]]:
    """Fetches tweets and user info for a Twitter user."""
    
    cached_data = cache.load("twitter", username)

    if cache.is_offline:
        if cached_data:
            logger.info(f"Offline mode: Using cached data for Twitter @{username}.")
            return cached_data
        else:
            logger.warning(f"Offline mode: No cache found for Twitter @{username}.")
            return {"timestamp": datetime.now(timezone.utc).isoformat(), "user_info": {}, "tweets": [], "media_analysis": [], "media_paths": []}

    if not force_refresh and cached_data and (datetime.now(timezone.utc) - get_sort_key(cached_data, "timestamp")) < timedelta(hours=CACHE_EXPIRY_HOURS):
        logger.info(f"Using recent cache for Twitter @{username}")
        return cached_data

    logger.info(f"Fetching Twitter data for @{username} (Force Refresh: {force_refresh})")
    since_id = None
    existing_tweets = []
    existing_media_analysis = []
    existing_media_paths = []
    user_info = None

    if not force_refresh and cached_data:
        existing_tweets = cached_data.get("tweets", [])
        if existing_tweets:
            since_id = existing_tweets[0].get("id")
        user_info = cached_data.get("user_info")
        existing_media_analysis = cached_data.get("media_analysis", [])
        existing_media_paths = cached_data.get("media_paths", [])

    try:
        if not user_info or force_refresh:
            user_response = client.get_user(
                username=username,
                user_fields=["created_at", "public_metrics", "profile_image_url", "verified", "description", "location"]
            )
            if not user_response or not user_response.data:
                raise UserNotFoundError(f"Twitter user @{username} not found.")
            user = user_response.data
            user_info = {
                "id": str(user.id), "name": user.name, "username": user.username,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "public_metrics": user.public_metrics, "profile_image_url": user.profile_image_url,
                "verified": user.verified, "description": user.description, "location": user.location
            }

        user_id = user_info["id"]
        new_tweets_data = []
        new_media_includes = {}
        all_users_from_includes = {}
        
        fetch_limit = INITIAL_FETCH_LIMIT if (force_refresh or not since_id) else INCREMENTAL_FETCH_LIMIT
        pagination_token = None
        tweets_fetch_count = 0

        while True:
            current_page_limit = min(fetch_limit - tweets_fetch_count, 100)
            if current_page_limit <= 0: break
            
            tweets_response = client.get_users_tweets(
                id=user_id, max_results=current_page_limit, since_id=since_id if not force_refresh else None,
                pagination_token=pagination_token,
                tweet_fields=["created_at", "public_metrics", "attachments", "entities", "conversation_id", "in_reply_to_user_id", "referenced_tweets"],
                expansions=["attachments.media_keys", "author_id", "in_reply_to_user_id", "referenced_tweets.id.author_id"],
                media_fields=["url", "preview_image_url", "type", "media_key", "width", "height", "alt_text"],
                user_fields=["username", "name", "id"]
            )
            
            if tweets_response.data:
                new_tweets_data.extend(tweets_response.data)
                tweets_fetch_count += len(tweets_response.data)
            
            if tweets_response.includes:
                if "media" in tweets_response.includes:
                    if "media" not in new_media_includes: new_media_includes["media"] = []
                    new_media_includes["media"].extend(tweets_response.includes["media"])
                if "users" in tweets_response.includes:
                    for user_obj in tweets_response.includes["users"]:
                        all_users_from_includes[str(user_obj.id)] = {"id": str(user_obj.id), "username": user_obj.username, "name": user_obj.name}
            
            pagination_token = tweets_response.meta.get("next_token")
            if not pagination_token or tweets_fetch_count >= fetch_limit:
                break
        
        logger.info(f"Fetched {tweets_fetch_count} total new tweets for @{username}.")
        processed_new_tweets = []
        newly_added_media_analysis = []
        newly_added_media_paths = set()
        all_media_objects = {m.media_key: m for m in new_media_includes.get("media", [])}

        auth_details = {"bearer_token": client.bearer_token}

        for tweet in new_tweets_data:
            media_items_for_tweet = []
            if tweet.attachments and "media_keys" in tweet.attachments:
                for media_key in tweet.attachments["media_keys"]:
                    media = all_media_objects.get(media_key)
                    if media:
                        url = media.url if media.type in ["photo", "gif"] and media.url else media.preview_image_url
                        if url:
                            media_path = download_media(cache.base_dir, url, cache.is_offline, "twitter", auth_details)
                            if media_path:
                                analysis = llm.analyze_image(media_path, f"Twitter user @{username}'s tweet (ID: {tweet.id})") if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else None
                                media_items_for_tweet.append({"type": media.type, "analysis": analysis, "url": url, "alt_text": media.alt_text, "local_path": str(media_path)})
                                if analysis: newly_added_media_analysis.append(analysis)
                                newly_added_media_paths.add(str(media_path))
            
            replied_to_user_info = all_users_from_includes.get(str(tweet.in_reply_to_user_id)) if tweet.in_reply_to_user_id else None
            
            tweet_data = {
                "id": str(tweet.id), "text": tweet.text, "created_at": tweet.created_at.isoformat(),
                "metrics": tweet.public_metrics, "entities_raw": tweet.entities,
                "mentions": [{"username": m["username"], "id": str(m["id"])} for m in (tweet.entities or {}).get("mentions", [])],
                "conversation_id": str(tweet.conversation_id),
                "in_reply_to_user_id": str(tweet.in_reply_to_user_id) if tweet.in_reply_to_user_id else None,
                "replied_to_user_info": replied_to_user_info,
                "referenced_tweets": [{"type": ref.type, "id": str(ref.id)} for ref in tweet.referenced_tweets] if tweet.referenced_tweets else [],
                "media": media_items_for_tweet
            }
            processed_new_tweets.append(tweet_data)

        combined_tweets = processed_new_tweets + existing_tweets
        unique_tweets = {t['id']: t for t in combined_tweets}
        final_tweets = sorted(list(unique_tweets.values()), key=lambda x: get_sort_key(x, "created_at"), reverse=True)
        
        final_media_analysis = sorted(list(set(newly_added_media_analysis + existing_media_analysis)))
        final_media_paths = sorted(list(newly_added_media_paths.union(existing_media_paths)))

        final_data = {
            "user_info": user_info, 
            "tweets": final_tweets[:MAX_CACHE_ITEMS], 
            "media_analysis": final_media_analysis, 
            "media_paths": final_media_paths[:MAX_CACHE_ITEMS*2]
        }
        cache.save("twitter", username, final_data)
        logger.info(f"Successfully updated Twitter cache for @{username}. Total tweets cached: {len(final_data['tweets'])}")
        return final_data

    except tweepy.TooManyRequests as e:
        # A more robust handler would parse the reset time from e.response.headers
        raise RateLimitExceededError("Twitter API rate limit exceeded.")
    except tweepy.errors.NotFound:
        raise UserNotFoundError(f"Twitter user @{username} not found.")
    except tweepy.errors.Forbidden as e:
        raise AccessForbiddenError(f"Access forbidden to @{username}'s tweets (protected/suspended). Reason: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching Twitter data for @{username}: {e}", exc_info=True)
        return None