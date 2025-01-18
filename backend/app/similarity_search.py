from datetime import datetime
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from services.question_checker import QuestionChecker
from timescale_vector import client

# Initialize VectorStore
vec = VectorStore()

# --------------------------------------------------------------
# Shipping question
# --------------------------------------------------------------

relevant_question = "วิศวสาขาเครื่องกล ภาคปกติ รอบ 1 โครงการเรียนล่วงหน้า มีเกณฑ์การรับอย่างไรบ้างคะ"

is_complete, feedback, major, round_, program, department = QuestionChecker.check_query(relevant_question)
print(f"Complete: {is_complete}")

if is_complete:
    results = vec.search(relevant_question, limit=3)

    response = Synthesizer.generate_response(question=relevant_question, context=results)

    print(f"\n{response.answer}")
    print("\nThought process:")
    for thought in response.thought_process:
        print(f"- {thought}")
    print(f"\nContext: {response.enough_context}")
    print("\nResults:")
    for idx, result in results.iterrows():
        print(f"Result {idx + 1}:\n{result['content']}\n")
        print(f"Distance: {result['distance']}")
else:
    print(f"feedback: {feedback}")

print(f"Extract from User Question using LLM Question Checker")
print(f"Major: {major}")
print(f"Round: {round_}")
print(f"Program: {program}")
print(f"Department: {department}")
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

# metadata_filter = {"major":"วศ.บ. สาขาวิชาวิศวกรรมเครื่องกล (นานาชาติ)" ,"admission_round": "2"}

# results = vec.search(relevant_question, limit=3, metadata_filter=metadata_filter)

# response = Synthesizer.generate_response(question=relevant_question, context=results)

# print(f"\n{response.answer}")
# print("\nThought process:")
# for thought in response.thought_process:
#     print(f"- {thought}")
# print(f"\nContext: {response.enough_context}")
# print("\nResults:")
# for idx, result in results.iterrows():
#     print(f"Result {idx + 1}:\n{result['content']}\n")
#     print(f"Distance: {result['distance']}")

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
