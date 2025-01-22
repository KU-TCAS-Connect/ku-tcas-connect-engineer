from datetime import datetime
from services.question_extraction import QuestionChecker, QueryCheckResponse
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from timescale_vector import client

# Initialize VectorStore
# vec = VectorStore()

# --------------------------------------------------------------
# Shipping question
# --------------------------------------------------------------

relevant_question = "วิศวกรรมไฟฟ้าเรียนล่วงหน้า รอบ2 ภาคปกติ มีเกณฑ์การรับเป็นอย่างไรบ้างและรับกี่คนคะ"
thought_process, major, round_, program, program_type = QuestionChecker.extract(relevant_question, QueryCheckResponse)

print(f"Extract from User Question using LLM Question Checker")
print(thought_process)
print(f"Major: {major}")
print(f"Round: {round_}")
print(f"Program: {program}")
print(f"Program Type: {program_type}")

# results = vec.search(relevant_question, limit=3)

# response = Synthesizer.generate_response(question=relevant_question, context=results)

# print(f"\n{response.answer}")
# print("\nThought process:")
# for thought in response.thought_process:
#     print(f"- {thought}")
# print(f"\nContext: {response.enough_context}")
# print("\nResults:")
# for idx, result in results.iterrows():
#     print(f"Result {idx + 1}:\n{result['content']}\n")

# --------------------------------------------------------------
# Irrelevant question
# --------------------------------------------------------------

# irrelevant_question = "วิศวกรรมไฟฟ้า มหาวิทยาลัยธรรมศาสตร์รังสิตมีเกณฑ์การรับอย่างไร"

# results = vec.search(irrelevant_question, limit=3)

# response = Synthesizer.generate_response(question=irrelevant_question, context=results)

# print(f"\n{response.answer}")
# print("\nThought process:")
# for thought in response.thought_process:
#     print(f"- {thought}")
# print(f"\nContext: {response.enough_context}")
# print("\nResults:")
# for idx, result in results.iterrows():
#     print(f"Result {idx + 1}:\n{result['content']}\n")

# # --------------------------------------------------------------
# # Metadata filtering
# # --------------------------------------------------------------

# metadata_filter = {"category": "Shipping"}

# results = vec.search(relevant_question, limit=3, metadata_filter=metadata_filter)

# response = Synthesizer.generate_response(question=relevant_question, context=results)

# print(f"\n{response.answer}")
# print("\nThought process:")
# for thought in response.thought_process:
#     print(f"- {thought}")
# print(f"\nContext: {response.enough_context}")

# # --------------------------------------------------------------
# # Advanced filtering using Predicates
# # --------------------------------------------------------------

# predicates = client.Predicates("category", "==", "Shipping")
# results = vec.search(relevant_question, limit=3, predicates=predicates)


# predicates = client.Predicates("category", "==", "Shipping") | client.Predicates(
#     "category", "==", "Services"
# )
# results = vec.search(relevant_question, limit=3, predicates=predicates)


# predicates = client.Predicates("category", "==", "Shipping") & client.Predicates(
#     "created_at", ">", "2024-09-01"
# )
# results = vec.search(relevant_question, limit=3, predicates=predicates)

# # --------------------------------------------------------------
# # Time-based filtering
# # --------------------------------------------------------------

# # September — Returning results
# time_range = (datetime(2024, 9, 1), datetime(2024, 9, 30))
# results = vec.search(relevant_question, limit=3, time_range=time_range)

# # August — Not returning any results
# time_range = (datetime(2024, 8, 1), datetime(2024, 8, 30))
# results = vec.search(relevant_question, limit=3, time_range=time_range)
