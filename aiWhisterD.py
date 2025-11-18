import asyncio
import json
import os
from typing import Literal

import httpx
import redis.asyncio as redis
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()


CAPTION_SYSTEM_PROMPT = """
As image analysis tool, concentrate on the physical description of a person in the picture. Ignore ephemeral attribures like mood, emotion, background, dress, actions 
and accessories. Try to sus out the gender, age, hair, skin tone, physique, ethnic and other physical descriptions that would help to uniquely identify the person. 
Consider if the person in the picture is human, or possibly a species from fantasy or science fiction, ie elves, orcs, ogres, and etc. 1-2 sentences is the sweet spot. 
Three sentences is pretty much the maximum length you should use unless there is something truly and outrageously unusual about the character.
Remember that you don’t need to describe every single thing about them: Pick out their most interesting and unique features. The description should be suitable to 
pass to a text to image generator.
"""

COMPARE_SYTEM_PROMPT = """
Consider hurp to be a function that compares two short character physical descriptions and tell me if they are in conflict or if
they generally agree on the description. Only if they do conflict, generate a plausible merge of the descriptions.
The generated description could merge some of the characteristics of each one. For example, if one describes an Asian person, and the other
a Caucasian person, offer an Eurasian description or mixed heritage. If attributes are in clear opposition, tall vs short, just pick one, 
or merge the two, for example if one is a prototypical sword and sorcery muscular male barbarian, in the manner of Conan for example,
and the other is a female then something like Red Sonja would be an acceptable plausible merge. 
"""


class Response(BaseModel):
    state: Literal["Congruent", "Conflict"] = Field(
        description="Do the inputs sematically match?"
    )
    explantion: str | None = Field(
        description="An explanation of how the inputs differ"
    )
    merge: str | None = Field(description="A plausible merge of the inputs")


async def process_job(client, job):
    print(f"Processing job {job}")
    image_file = job["image_file"]
    with open(image_file, "rb") as f:
        image_bytes = f.read()
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            "Describe the person in the picture.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=CAPTION_SYSTEM_PROMPT,
        ),
    )
    if response is None:
        return
    desc1 = response.text
    print(desc1)
    if job["current_description"] != "":
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=f"hurp({job['current_description'], desc1})",
            config=types.GenerateContentConfig(
                system_instruction=COMPARE_SYTEM_PROMPT,
                response_mime_type="application/json",
                response_json_schema=Response.model_json_schema(),
            ),
        )
        if response is None:
            return
        struct = Response.model_validate_json(response.text)
    else:
        struct = Response(state="Conflict", explantion=None, merge=desc1)
    print(struct)
    if struct.state == "Conflict":
        async with httpx.AsyncClient() as http_client:
            response = await http_client.put(
                "http://127.0.0.1:5000/api/v1/ai/work/caption/complete",
                json={
                    "character_id": job["character_id"],
                    "explanation": struct.explantion,
                    "merge": struct.merge,
                },
            )


async def main():
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    while True:
        raw = await redis_client.blpop(["work_queue"], 0) #type: ignore
        if raw:
            _, value = raw
            job = json.loads(value)
            asyncio.create_task(process_job(client, job))


if __name__ == "__main__":
    asyncio.run(main())
