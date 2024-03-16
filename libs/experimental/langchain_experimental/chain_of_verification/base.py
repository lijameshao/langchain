"""Chain of verification https://arxiv.org/abs/2309.11495"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Extra, Field
from langchain.schema import StrOutputParser
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import Runnable, RunnablePassthrough

from langchain_experimental.chain_of_verification.prompt import (
    BASELINE_RESPONSE_PROMPT,
    EXECUTE_VERIFICATIONS_PROMPT,
    PLAN_VERIFICATIONS_PROMPT,
    REVISED_RESPONSE_PROMPT,
)


class PlanVerificationsOutputModel(BaseModel):
    query: str = Field(description="The user's query")
    baseline_response: str = Field(description="The response to the user's query")
    facts_and_verification_questions: dict[str, str] = Field(
        description="Facts (as the dictionary keys) extracted from the response and"
        " verification questions related to the query (as the dictionary values)"
    )


class CoVeChain(Chain):
    """Chain of Verification (CoVe)

    Example:
        .. code-block:: python

            from langchain.llms import OpenAI
            from langchain_experimental.chain_of_verification import CoVeChain
            llm = OpenAI(temperature=0)
            cove_chain = CoVeChain.from_llm(llm)
            cove_chain(
                {"query": "Name some politicians who were born in New York?"}
            )

            # Using runnable invoke() method and callbacks
            from langchain.callbacks import StdOutCallbackHandler

            handler = StdOutCallbackHandler()
            cove_chain.invoke(
                {"query": "Name some politicians who were born in New York?"},
                {"callbacks": [handler]},
            )
    """

    cove_chain: Runnable
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            run_manager.on_text("Running Chain of Verification...\n")
        query = inputs[self.input_key]
        output = self.cove_chain.invoke(query)

        # Log intermediate steps
        # TODO: find a way to log during the execution of the chain
        if run_manager:
            run_manager.on_text(f"Baseline response: {output['baseline_response']}\n\n")
            run_manager.on_text(
                f"Verification questions/answers: \n{output['verified_responses']}\n\n"
            )
        return {self.output_key: output["revised_response"]}

    @property
    def _chain_type(self) -> str:
        return "cove_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        baseline_response_prompt: PromptTemplate = BASELINE_RESPONSE_PROMPT,
        plan_verifications_prompt: PromptTemplate = PLAN_VERIFICATIONS_PROMPT,
        execute_verifications_prompt: PromptTemplate = EXECUTE_VERIFICATIONS_PROMPT,
        revised_response_prompt: PromptTemplate = REVISED_RESPONSE_PROMPT,
        **kwargs: Any,
    ) -> CoVeChain:
        plan_verifications_output_parser = PydanticOutputParser(
            pydantic_object=PlanVerificationsOutputModel
        )
        # Answer the user's query
        baseline_response_chain = (
            baseline_response_prompt | llm | StrOutputParser()
        ).with_config(run_name="Baseline Response")

        # Generate verification questions
        plan_verifications_response = (
            plan_verifications_prompt
            | llm
            | plan_verifications_output_parser
            | (lambda output: list(output.facts_and_verification_questions.values()))
        ).with_config(run_name="Plan Verifications")

        # Execute verification questions in parallel
        verification_chain = (
            (
                lambda x: [
                    {"verification_question": question}
                    for question in x["verification_questions"]
                ]
            )
            | (execute_verifications_prompt | llm | StrOutputParser()).map()
        ).with_config(run_name="Execute Verifications in parallel")

        # Based on the verification questions, revise the final response
        revise_response_chain = (
            revised_response_prompt | llm | StrOutputParser()
        ).with_config(run_name="Revise Response")

        cove_chain = (
            {"query": RunnablePassthrough()}
            | RunnablePassthrough.assign(baseline_response=baseline_response_chain)
            | RunnablePassthrough.assign(
                verification_questions=plan_verifications_response
            )
            | RunnablePassthrough.assign(verification_responses=verification_chain)
            | RunnablePassthrough.assign(
                verified_responses=lambda x: "\n\n".join(
                    [
                        f"Question: {question}\nAnswer: {answer}\n\n"
                        for question, answer in zip(
                            x["verification_questions"],
                            x["verification_responses"],
                        )
                    ]
                )
            )
            | RunnablePassthrough.assign(revised_response=revise_response_chain)
        )
        return cls(
            cove_chain=cove_chain,
            **kwargs,
        )
