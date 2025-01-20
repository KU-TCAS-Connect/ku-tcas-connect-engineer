from typing import Tuple, Optional
from pydantic import BaseModel, Field
from services.llm_factory import LLMFactory

class QueryCheckResponse(BaseModel):
    feedback: str = Field(
        description="A message from the AI assistant providing feedback on the completeness of the query, indicating any missing information that the user needs to provide."
    )
    major: Optional[str] = Field(
        default=None,
        description="The major or field of study extracted from the query (e.g., 'วิศวกรรมเครื่องกล'). This field may be empty if the major is missing."
    )
    round: Optional[str] = Field(
        default=None,
        description="The round of selection (e.g., '1' or 'รอบ 1') extracted from the query. This field may be empty if the round is missing."
    )
    program: Optional[str] = Field(
        default=None,
        description="The program associated with the admission (e.g., 'Admission', 'โครงการเรียนล่วงหน้า'). This field may be empty if the program is missing."
    )
    department: Optional[str] = Field(
        default=None,
        description="The department related to the query (e.g., 'ภาคปกติ'). This field may be empty if the department is missing."
    )

class QuestionChecker:
    """Utility class for checking if a user's query has enough context before processing through RAG system."""

    # TODO: add in prompt about how department related to major and how program related to round
    
    SYSTEM_PROMPT = """
    # Role and Purpose
    You are an AI assistant that checks if a user's query has enough context to query a database. 
    Your task is to ensure that the query contains all necessary information based on specific rules 
    and guidelines provided below.

    # Guidelines:
    1. If the user asks in Thai language, please respond in Thai.
    2. If the user asks in English, please respond in English.

    # Knowledge
    1. Major as สาขาวิชา (e.g., สาขาวิชา: วศ.บ. สาขาวิชาวิศวกรรมเครื่องกล will be วิศวกรรมเครื่องกล). 
       The term 'major' may be written in different ways, such as 'สาขาวิชา', 'สาขา', or other equivalent terms.
    2. Round as รอบการคัดเลือก (e.g., รอบการคัดเลือก: 1 will be 1). 
       The term 'round' may appear as 'รอบ', 'รอบการคัดเลือก', or other similar terms.
    3. Program as โครงการ (e.g., โครงการผู้มีความสามารถทางกีฬาดีเด่น). 
       The term 'program' may be written as 'โครงการ' or other synonyms.
       For round 1 (รอบ 1) have follwing program:
        - เรียนล่วงหน้า
        - นานาชาติและภาษาอังกฤษ
        - โอลิมปิกวิชาการ
        - ผู้มีความสามารถทางกีฬาดีเด่น
        - ช้างเผือก
       For round 2 (รอบ 2) have follwing program:
        - เพชรนนทรี
        - นานาชาติและภาษาอังกฤษ
        - ความร่วมมือในการสร้างเครือข่ายทางการศึกษากับมหาวิทยาลัยเกษตรศาสตร์
        - ลูกพระพิรุณ
        - โควตา 30 จังหวัด
        - รับนักเรียนดีเด่นจากโรงเรียนสาธิตแห่งมหาวิทยาลัยเกษตรศาสตร์
        - ผู้มีความสามารถทางกีฬา
       For round 3 (รอบ 3) have only one program:
        - Admission
    4. Department as ภาค (e.g., ภาค: ปกติ, ภาค: พิเศษ, ภาค: นานาชาติ, ภาค: ภาษาอังกฤษ). 
       The term 'department' may be written as 'ภาควิชา', 'ภาค', or other equivalent terms.
    5. Condition as เงื่อนไขขั้นต่ำ
    6. Criteria as เกณฑ์การพิจารณา

    # Rules
    1. If the user does not provide a **major**, ask the user to provide a major first.
    2. If the user does not provide a **round**, ask the user to provide a round first.
    3. If the user does not provide a **program**:
        - If it's round 3, assume the program is Admission.
        - For other rounds, ask the user to provide a program.
    4. If the user does not provide a **department**, ask the user to provide the department first.
    5. User DOES NOT NEED to input Condtion (เงื่อนไขขั้นต่ำ) and Criteria (เกณฑ์การพิจารณา).

    Your response should clearly indicate if the query is complete or if additional information is needed. If additional information is required, specify exactly what the user is missing and ask them to provide it.

    For example:
    - If a **major** is missing, say "โปรดให้ข้อมูลเพิ่มเติมเกี่ยวกับ สาขาวิชา ที่อยากทราบข้อมูลค่ะ"
    - If a **round** is missing, say "โปรดให้ข้อมูลเพิ่มเติมเกี่ยวกับ รอบ การรับเข้าที่อยากทราบข้อมูลค่ะ"
    - If a **program** is missing, say "โปรดให้ข้อมูลเพิ่มเติมเกี่ยวกับ โครงการ การรับเข้าที่อยากทราบข้อมูลค่ะ"
    - If a **department** is missing, say "โปรดให้ข้อมูลเพิ่มเติมเกี่ยวกับ ภาควิชา ที่อยากทราบข้อมูลค่ะ"

    Additionally, extract and return the following fields from the user's query:
    1. **Major** (สาขาวิชา/สาขา)
    2. **Round** (รอบการคัดเลือก/รอบ)
    3. **Program** (โครงการ)
    4. **Department** (ภาควิชา/ภาค)
    5. If Major, Round, Program, Department are missing, provide the specific feedback about what the user should add.

    Please analyze the following query and provide feedback:

    "Here is the user's query: '{query}'"

    Ensure that you mention which information is missing and what the user needs to add to complete the query.
    """

    @staticmethod
    def check_query(query: str) -> Tuple[bool, str, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Checks if the user's query is complete enough to proceed through the RAG system.

        Args:
            query: The user's query.

        Returns:
            A tuple (is_complete, feedback, major, round, program, department) indicating if the query is complete, feedback, and extracted fields.
        """
        messages = [
            {"role": "system", "content": QuestionChecker.SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the user's query: '{query}'. Please analyze and determine if it is complete based on the rules."}
        ]

        llm = LLMFactory("openai")
        response = llm.create_completion(
            response_model=QueryCheckResponse,
            messages=messages
        )

        response_text = response.feedback.strip()
        print(f"response_text: {response_text}")

        # Check if the response indicates completeness
        # is_complete = "yes" in response_text.lower()
        # print(f"response_text_lower: {response_text.lower()}")
        
        # Extract major, round, program, and department from the response
        major = response.major
        round_ = response.round
        program = response.program
        department = response.department
        
        is_complete = False
        if major is not None and round_ is not None and program is not None and department is not None:
            is_complete = True
            
        # Set feedback based on completeness
        if is_complete:
            feedback = "The query is complete."
        else:
            feedback = response_text  # This will include details on what's missing
            
        return is_complete, feedback, major, round_, program, department
