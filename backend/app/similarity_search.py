from datetime import datetime
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from services.question_checker import QuestionChecker
from timescale_vector import client
import pythainlp

def tokenize_and_search(query, text):
    query_tokens = pythainlp.word_tokenize(query)
    text_tokens = pythainlp.word_tokenize(text)
    matches = [token for token in query_tokens if token in text_tokens]
    return matches

# Initialize VectorStore
vec = VectorStore()

# --------------------------------------------------------------
# Relevant question
# --------------------------------------------------------------

# # relevant_question = "วิศวสาขาเครื่องกล ภาคปกติ รอบ 1 โครงการเรียนล่วงหน้า มีเกณฑ์การรับอย่างไรบ้างคะ"
# relevant_question = "วิศวคอมพิวเตอร์ ภาคปกติ รอบ 1 เรียนล่วงหน้า มีเกณฑ์การรับอย่างไรบ้างคะ"

# is_complete, feedback, major, round_, program, department = QuestionChecker.check_query(relevant_question)
# print(f"Complete: {is_complete}")

# # keyword = vec.keyword_search(relevant_question)
# # print(keyword)


# if is_complete:
#     results = vec.search(relevant_question, limit=3)

#     document_from_db_before_filter = """
#     Retrieved documents:
#     """
#     for idx, result in results.iterrows():
#         document_from_db_before_filter += f"Retrieved documents {idx + 1}:\n{result['content']}\n"
#         document_from_db_before_filter += f"\n"

#     print("Document from db before filter", document_from_db_before_filter)

#     response = Synthesizer.generate_response(question=relevant_question, context=results, context_before_filter=document_from_db_before_filter)

#     print(f"\n{response.answer}")
#     print("\nThought process:")
#     for thought in response.thought_process:
#         print(f"- {thought}")
#     print(f"\nContext: {response.enough_context}")
#     print("\nResults:")
#     for idx, result in results.iterrows():
#         print(f"Result {idx + 1}:\n{result['content']}\n")
#         print(f"Distance: {result['distance']}")
#         # print("test", tokenize_and_search(query=relevant_question, text=result['content']))
# else:
#     print(f"feedback: {feedback}")


# print(f"Extract from User Question using LLM Question Checker")
# print(f"Major: {major}")
# print(f"Round: {round_}")
# print(f"Program: {program}")
# print(f"Department: {department}")


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

# relevant_question = "วิศวสาขาเครื่องกล ภาคปกติ รอบ 1 โครงการเรียนล่วงหน้า มีเกณฑ์การรับอย่างไรบ้างคะ"
relevant_question = "วิศวคอมพิวเตอร์ ภาคปกติ รอบ 1 เรียนล่วงหน้า มีเกณฑ์การรับอย่างไรบ้างคะ"

is_complete, feedback, major, round_, program, department = QuestionChecker.check_query(relevant_question)
print(f"Complete: {is_complete}")

print(f"Extract from User Question using LLM Question Checker")
print(f"Major: {major}")
print(f"Round: {round_}")
print(f"Program: {program}")
print(f"Department: {department}")


metadata_filter = {"major":"วศ.บ. สาขาวิชาวิศวกรรมเครื่องกล (นานาชาติ)" ,"admission_round": f"{round_}"}
# metadata_filter = {"admission_program": f"โครงการ{program}" ,"admission_round": f"{round_}"}

results = vec.search(relevant_question, limit=3, metadata_filter=metadata_filter)


# keyword = vec.keyword_search(relevant_question)
# print(keyword)

if is_complete:
    results = vec.search(relevant_question, limit=3)

    document_from_db_before_filter = """
    Retrieved documents:
    """
    for idx, result in results.iterrows():
        document_from_db_before_filter += f"Retrieved documents {idx + 1}:\n{result['content']}\n"
        document_from_db_before_filter += f"\n"

    print("Document from db before filter", document_from_db_before_filter)

    response = Synthesizer.generate_response(question=relevant_question, context=results, context_before_filter=document_from_db_before_filter)

    print(f"\n{response.answer}")
    print("\nThought process:")
    for thought in response.thought_process:
        print(f"- {thought}")
    print(f"\nContext: {response.enough_context}")
    print("\nResults:")
    for idx, result in results.iterrows():
        print(f"Result {idx + 1}:\n{result['content']}\n")
        print(f"Distance: {result['distance']}")
        # print("test", tokenize_and_search(query=relevant_question, text=result['content']))
else:
    print(f"feedback: {feedback}")

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
