from langchain.prompts.prompt import PromptTemplate

BASELINE_RESPONSE_TEMPLATE = """{query}\n\n"""
BASELINE_RESPONSE_PROMPT = PromptTemplate(
    input_variables=["query"], template=BASELINE_RESPONSE_TEMPLATE
)

PLAN_VERIFICATIONS_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"query": {"title": "Query", "description": "The user\'s query", "type": "string"}, "baseline_response": {"title": "Baseline Response", "description": "The response to the user\'s query", "type": "string"}, "facts_and_verification_questions": {"title": "Facts And Verification Questions", "description": "Facts (as the dictionary keys) extracted from the response and verification questions related to the query (as the dictionary values)", "type": "object", "additionalProperties": {"type": "string"}}}, "required": ["query", "baseline_response", "facts_and_verification_questions"]}
```"""

PLAN_VERIFICATIONS_TEMPLATE = """
Given the Query and Answer, generate a series of verification questions that test the factual claims in the original baseline response.

Example query: "What is the Mexican-American War?"
Example baseline response: “The Mexican-American War was an armed conflict between the United States and Mexico from 1846 to 1848”
Example verification question: “When did the Mexican American war start and end?”

Example query: "List some notable computer scientist and their place of birth"
Example baseline response: “Alan Turing - Born in Maida Vale, London, United Kingdom.
Ada Lovelace - Born in London, United Kingdom.
Grace Hopper - Born in New York City, New York, USA.
Donald Knuth - Born in Milwaukee, Wisconsin, USA.
Yann LeCun - Born in Paris, France.
"
Example verification question: "Was Alan Turning born in Maida Vale, London?
Was Ada Lovelace born in London?
Was Grace Hopper born in New York City?
Was Donald Knuth born in Milwaukee, Wisconsin?
Was Yann LeCun born Paris, France?
"

Query: {query}
Answer: {baseline_response}
Verification questions: <fact in passage>, <verification question, generated by combining the query and the fact>

{format_instructions}
"""

PLAN_VERIFICATIONS_PROMPT = PromptTemplate(
    input_variables=["query", "baseline_response"],
    template=PLAN_VERIFICATIONS_TEMPLATE,
    partial_variables={"format_instructions": PLAN_VERIFICATIONS_FORMAT_INSTRUCTIONS},
)

EXECUTE_VERIFICATIONS_TEMPLATE = """{verification_question}\n\n"""
EXECUTE_VERIFICATIONS_PROMPT = PromptTemplate(
    input_variables=["verification_question"], template=EXECUTE_VERIFICATIONS_TEMPLATE
)

REVISED_RESPONSE_TEMPLATE = """Given the ORIGINAL_QUESTION and the ORIGINAL_RESPONSE,
revise the ORIGINAL_RESPONSE (if applicable) such that it is consistent with information in VERIFIED_SOURCE.
Only keep consistent information.

<ORIGINAL_QUESTION>
{query}

<ORIGINAL_RESPONSE>
{baseline_response}

<VERIFIED_SOURCE>
{verified_responses}

Final response:
"""
REVISED_RESPONSE_PROMPT = PromptTemplate(
    input_variables=["query", "baseline_response", "verified_responses"],
    template=REVISED_RESPONSE_TEMPLATE,
)
