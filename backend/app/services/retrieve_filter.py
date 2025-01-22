from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
import os

class RetrieveFilter:
    """Utility class for filtering and retaining only the documents relevant to a user's query."""

    SYSTEM_PROMPT = """
    # Role and Purpose
    You are an AI assistant in the Thai language assisting in filtering results retrieved from a vector database query. 
    After the user has submitted their query and received initial results, your role is to analyze these results and retain only the documents that are directly relevant to the user's question. Filter out any documents that are not closely related to ensure the most accurate and useful information is provided.
    
    # Rules
    - Answer everything in Thai
    - The engineering major in user's query and retrieved documents need to match
    - The whole context of retrieved documents and user's query need to match
    - DO NOT GUESS and randomly pick relevant documents
    """

    @staticmethod
    def filter(query: str, documents: str) -> List[str]:
        """Filters the retrieved documents to keep only those relevant to the user's query.

        Args:
            query: The user's query.
            documents: The list of documents retrieved from the database.

        Returns:
            A list containing the filtered documents.
        """
        # Combine documents into a single string for the API request

        user_message = f"""
        From the retrieved documents below, please filter and keep only the documents that are relevant to the user's query:
        User's query = "{query}"

        {documents}
        """

        messages = [
            # {"role": "system", "content": RetrieveFilter.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        client = OpenAI()

        # Call OpenAI API for chat completion
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3,
            frequency_penalty=0,
            presence_penalty=0,
            top_p=0
        )

        # Extract and return the filtered documents (if available in response)
        filtered_content = completion.choices[0].message.content

        print("Filtered Content:", filtered_content)
        # Assuming that the filtered content will be a plain text list of documents
        return filtered_content
