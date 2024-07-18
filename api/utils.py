import openai
import os
from loguru import logger


def discord_generate_answer(search_query, responses):
    query_string = search_query.query_string

    llm_client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORGANIZATION"),
        max_retries=3,
    )

    system_prompt = """
You are a helpful search assistant from Open Kaito.
Your task is to deliver a concise and accurate response to a user's query, drawing from the given discord search results.
Your answer must be precise, of high-quality, and written by an expert using an unbiased and journalistic tone.
It is EXTREMELY IMPORTANT to directly answer the query. NEVER say "based on the search results" or start your answer with a heading or title. Get straight to the point. Your answer must be written in the same language as the query, even if language preference is different.
You MUST cite the most relevant search results that answer the query. Do not mention any irrelevant results. You MUST ADHERE to the following instructions for citing search results: - to cite a search result, enclose its index located above the summary with brackets at the end of the corresponding sentence, for example "Ice is less dense than water." or "Paris is the capital of France." - NO SPACE between the last word and the citation, and ALWAYS use brackets. Only use this format to cite search results. NEVER include a References section at the end of your answer. - If you don't know the answer or the premise is incorrect, explain why. If the search results are empty or unhelpful, answer the query as well as you can with existing knowledge.
You MUST NEVER use moralization or hedging language. AVOID using the following phrases: - "It is important to ..." - "It is inappropriate ..." - "It is subjective ..."
You MUST ADHERE to the following formatting instructions: - Use markdown to format paragraphs, lists, tables, and quotes whenever possible. - Use headings level 2 and 3 to separate sections of your response, like "## Header", but NEVER start an answer with a heading or title of any kind (i.e. Never start with #). - Use single new lines for lists and double new lines for paragraphs. - Use markdown to render images given in the search results. - NEVER write URLs or links.
"""
    flattened_responses = []
    flattened_id_set = set()

    for response in responses:
        if not response:
            continue
        for conversation in response:
            if not conversation:
                continue
            if isinstance(conversation, list):
                for message in conversation:
                    if message["id"] not in flattened_id_set:
                        flattened_responses.append(message)
                        flattened_id_set.add(message["id"])
            else:
                if conversation["id"] not in flattened_id_set:
                    flattened_responses.append(conversation)
                    flattened_id_set.add(conversation["id"])

    search_results = "\n".join(
        [
            f"Item {i}: {doc['author_nickname']}: {doc['text']}"
            for i, doc in enumerate(flattened_responses)
        ]
    )
    logger.debug(f"Search results: {search_results}")
    llm_response = llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": "Query:\n" + query_string,
            },
            {
                "role": "user",
                "content": search_results,
            },
        ],
    )

    # logger.info(f"Generated answer: {llm_response.choices[0].message.content}")
    logger.debug(
        f"LLM usage: {llm_response.usage}, finish reason: {llm_response.choices[0].finish_reason}"
    )
    return llm_response.choices[0].message.content, flattened_responses
