#!/usr/bin/env python3
"""
Quick manual test client.
Usage: python scripts/test_client.py "Your research question here"
"""
import asyncio
import sys
import httpx

BASE_URL = "http://localhost:8000/api/v1"
API_KEY = "dev-key-replace-in-production"
HEADERS = {"X-API-Key": API_KEY}


async def run(query: str, depth: str = "standard") -> None:
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Submit job
        print(f"\n🔍 Submitting research: {query!r} [{depth}]\n")
        resp = await client.post(
            f"{BASE_URL}/research",
            json={"query": query, "depth": depth},
            headers=HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()
        job_id = data["job_id"]
        print(f"✅ Job accepted: {job_id}\n")

        # Poll until done
        print("⏳ Polling for result...")
        while True:
            await asyncio.sleep(3)
            poll = await client.get(f"{BASE_URL}/research/{job_id}", headers=HEADERS)
            poll.raise_for_status()
            job = poll.json()
            status = job["status"]
            print(f"   Status: {status}")

            if status == "completed":
                result = job["result"]
                print(f"\n{'='*60}")
                print(f"📋 SUMMARY:\n{result['summary']}\n")
                print(f"🔑 KEY FINDINGS:")
                for i, finding in enumerate(result["key_findings"], 1):
                    print(f"  {i}. {finding}")
                print(f"\n📚 SOURCES ({result['sources_scraped']} scraped):")
                for c in result["citations"][:3]:
                    print(f"  • {c['title']} — {c['url']}")
                print(f"\n🎯 Confidence: {result['confidence_score']:.0%}")
                break

            elif status == "failed":
                print(f"❌ Job failed: {job.get('error_message')}")
                break


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the latest advancements in quantum computing in 2025?"
    asyncio.run(run(query))
