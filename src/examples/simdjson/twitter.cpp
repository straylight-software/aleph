// src/examples/simdjson/twitter.cpp
//
// simdjson 4.2.4 demo - SIMD-accelerated JSON parsing at 4+ GB/s
//
// Demonstrates:
//   - On-demand parsing (DOM-free, lazy evaluation)
//   - Nested object traversal
//   - Array iteration
//   - Error handling without exceptions
//   - Performance measurement
//
// This example parses a mock Twitter API response structure,
// extracting user info and tweet text from nested JSON.

#include <simdjson.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace straylight::examples {

// ════════════════════════════════════════════════════════════════════════════════
// Mock Twitter API response - realistic nested structure
// ════════════════════════════════════════════════════════════════════════════════

constexpr std::string_view kTwitterResponse = R"({
  "data": [
    {
      "id": "1445078208190291968",
      "text": "The matrix has its roots in primitive arcade games",
      "author_id": "12345",
      "created_at": "2026-01-26T12:00:00.000Z",
      "public_metrics": {
        "retweet_count": 42,
        "reply_count": 7,
        "like_count": 256,
        "quote_count": 3
      },
      "entities": {
        "hashtags": [
          {"start": 0, "end": 6, "tag": "matrix"},
          {"start": 30, "end": 36, "tag": "arcade"}
        ],
        "urls": []
      }
    },
    {
      "id": "1445078208190291969",
      "text": "Cyberspace. A consensual hallucination experienced daily by billions",
      "author_id": "12346",
      "created_at": "2026-01-26T12:01:00.000Z",
      "public_metrics": {
        "retweet_count": 128,
        "reply_count": 15,
        "like_count": 512,
        "quote_count": 8
      },
      "entities": {
        "hashtags": [
          {"start": 0, "end": 10, "tag": "cyberspace"}
        ],
        "urls": [
          {"start": 50, "end": 73, "url": "https://straylight.ai", "expanded_url": "https://straylight.ai/neuromancer"}
        ]
      }
    },
    {
      "id": "1445078208190291970",
      "text": "The sky above the port was the color of television, tuned to a dead channel",
      "author_id": "12347",
      "created_at": "2026-01-26T12:02:00.000Z",
      "public_metrics": {
        "retweet_count": 1024,
        "reply_count": 89,
        "like_count": 4096,
        "quote_count": 42
      },
      "entities": {
        "hashtags": [],
        "urls": []
      }
    }
  ],
  "includes": {
    "users": [
      {"id": "12345", "name": "Case", "username": "case_cowboy", "verified": false},
      {"id": "12346", "name": "Molly", "username": "razorgirl", "verified": true},
      {"id": "12347", "name": "Wintermute", "username": "wintermute_ai", "verified": true}
    ]
  },
  "meta": {
    "result_count": 3,
    "newest_id": "1445078208190291970",
    "oldest_id": "1445078208190291968"
  }
})";

// ════════════════════════════════════════════════════════════════════════════════
// Tweet structure for extraction
// ════════════════════════════════════════════════════════════════════════════════

struct Tweet {
  std::string id;
  std::string text;
  std::string author_id;
  int64_t likes{0};
  int64_t retweets{0};
  std::vector<std::string> hashtags;
};

struct User {
  std::string id;
  std::string name;
  std::string username;
  bool verified{false};
};

// ════════════════════════════════════════════════════════════════════════════════
// Parse tweets using simdjson on-demand API
// ════════════════════════════════════════════════════════════════════════════════

auto parse_tweets(std::string_view json) -> std::vector<Tweet> {
  std::vector<Tweet> tweets;

  simdjson::ondemand::parser parser;
  simdjson::padded_string padded(json);
  
  auto doc = parser.iterate(padded);
  
  // Navigate to data array
  auto data = doc["data"];
  
  for (auto tweet_obj : data.get_array()) {
    Tweet tweet;
    
    // Extract basic fields - simdjson uses implicit conversion
    std::string_view id_sv;
    tweet_obj["id"].get_string().get(id_sv);
    tweet.id = std::string(id_sv);
    
    std::string_view text_sv;
    tweet_obj["text"].get_string().get(text_sv);
    tweet.text = std::string(text_sv);
    
    std::string_view author_sv;
    tweet_obj["author_id"].get_string().get(author_sv);
    tweet.author_id = std::string(author_sv);
    
    // Extract metrics
    auto metrics = tweet_obj["public_metrics"];
    metrics["like_count"].get_int64().get(tweet.likes);
    metrics["retweet_count"].get_int64().get(tweet.retweets);
    
    // Extract hashtags
    auto entities = tweet_obj["entities"];
    auto hashtags = entities["hashtags"];
    for (auto ht : hashtags.get_array()) {
      std::string_view tag_sv;
      ht["tag"].get_string().get(tag_sv);
      tweet.hashtags.push_back(std::string(tag_sv));
    }
    
    tweets.push_back(std::move(tweet));
  }
  
  return tweets;
}

// ════════════════════════════════════════════════════════════════════════════════
// Parse users
// ════════════════════════════════════════════════════════════════════════════════

auto parse_users(std::string_view json) -> std::vector<User> {
  std::vector<User> users;

  simdjson::ondemand::parser parser;
  simdjson::padded_string padded(json);
  
  auto doc = parser.iterate(padded);
  auto includes = doc["includes"];
  auto users_array = includes["users"];
  
  for (auto user_obj : users_array.get_array()) {
    User user;
    
    std::string_view sv;
    user_obj["id"].get_string().get(sv);
    user.id = std::string(sv);
    
    user_obj["name"].get_string().get(sv);
    user.name = std::string(sv);
    
    user_obj["username"].get_string().get(sv);
    user.username = std::string(sv);
    
    user_obj["verified"].get_bool().get(user.verified);
    
    users.push_back(std::move(user));
  }
  
  return users;
}

// ════════════════════════════════════════════════════════════════════════════════
// Benchmark: parse the same JSON many times
// ════════════════════════════════════════════════════════════════════════════════

auto benchmark_parsing(std::string_view json, int iterations) -> double {
  simdjson::ondemand::parser parser;
  simdjson::padded_string padded(json);
  
  auto start = std::chrono::high_resolution_clock::now();
  
  int64_t total_likes = 0;
  for (int i = 0; i < iterations; ++i) {
    auto doc = parser.iterate(padded);
    auto data = doc["data"];
    for (auto tweet : data.get_array()) {
      auto metrics = tweet["public_metrics"];
      int64_t likes = 0;
      metrics["like_count"].get_int64().get(likes);
      total_likes += likes;
    }
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(end - start).count();
  
  // Prevent optimization from removing the loop
  if (total_likes == 0) {
    std::printf("unexpected zero likes\n");
  }
  
  return duration;
}

// ════════════════════════════════════════════════════════════════════════════════
// Main demo
// ════════════════════════════════════════════════════════════════════════════════

auto implementation() -> int {
  std::printf("════════════════════════════════════════════════════════════\n");
  std::printf("  simdjson %s - SIMD-accelerated JSON parsing\n", 
              SIMDJSON_VERSION);
  std::printf("════════════════════════════════════════════════════════════\n\n");
  
  // Show implementation info
  std::printf("Implementation: %s\n", simdjson::get_active_implementation()->name().data());
  std::printf("Description: %s\n\n", simdjson::get_active_implementation()->description().data());
  
  // Parse tweets
  std::printf("Parsing Twitter API response (%zu bytes)...\n\n", kTwitterResponse.size());
  
  auto tweets = parse_tweets(kTwitterResponse);
  auto users = parse_users(kTwitterResponse);
  
  // Display results
  std::printf("Found %zu tweets:\n", tweets.size());
  std::printf("────────────────────────────────────────────────────────────\n");
  
  for (const auto& tweet : tweets) {
    // Find author
    std::string author_name = "unknown";
    for (const auto& user : users) {
      if (user.id == tweet.author_id) {
        author_name = user.name;
        if (user.verified) {
          author_name += " [verified]";
        }
        break;
      }
    }
    
    std::printf("\n@%s:\n", author_name.c_str());
    std::printf("  \"%s\"\n", tweet.text.c_str());
    std::printf("  likes: %ld  retweets: %ld", tweet.likes, tweet.retweets);
    
    if (!tweet.hashtags.empty()) {
      std::printf("  tags: ");
      for (size_t i = 0; i < tweet.hashtags.size(); ++i) {
        std::printf("#%s", tweet.hashtags[i].c_str());
        if (i < tweet.hashtags.size() - 1) std::printf(", ");
      }
    }
    std::printf("\n");
  }
  
  // Benchmark
  std::printf("\n════════════════════════════════════════════════════════════\n");
  std::printf("  Performance benchmark\n");
  std::printf("════════════════════════════════════════════════════════════\n\n");
  
  constexpr int kIterations = 100000;
  double ms = benchmark_parsing(kTwitterResponse, kIterations);
  
  double bytes_processed = static_cast<double>(kTwitterResponse.size()) * kIterations;
  double gb_per_sec = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
  
  std::printf("Parsed %d iterations in %.2f ms\n", kIterations, ms);
  std::printf("Throughput: %.2f GB/s\n", gb_per_sec);
  std::printf("Per-parse: %.3f microseconds\n\n", (ms * 1000.0) / kIterations);
  
  std::printf("════════════════════════════════════════════════════════════\n");
  std::printf("  simdjson: parsing JSON at the speed of your CPU\n");
  std::printf("════════════════════════════════════════════════════════════\n");
  
  return 0;
}

} // namespace straylight::examples

auto main() -> int {
  return straylight::examples::implementation();
}
