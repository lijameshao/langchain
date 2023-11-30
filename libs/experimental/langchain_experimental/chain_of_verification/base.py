"""Chain of verification https://arxiv.org/abs/2309.11495"""
from __future__ import annotations

from operator import itemgetter
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, Extra
from langchain.schema import StrOutputParser
from langchain.schema.language_model import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain_experimental.chain_of_verification.prompt import (
    BASELINE_RESPONSE_PROMPT,
    PLAN_VERIFICATIONS_PROMPT,
    EXECUTE_VERIFICATIONS_PROMPT,
    REVISED_RESPONSE_PROMPT,
)


class PlanVerificationsOutputModel(BaseModel):
    query: str = Field(description="The user's query")
    baseline_response: str = Field(description="The response to the user's query")
    facts_and_verification_questions: dict[str, str] = Field(
        description="Facts (as the dictionary keys) extracted from the response and verification questions related to the query (as the dictionary values)"
    )


class CoVeChain(Chain):
    """Chain of Verification (CoVe)

    Example:
        .. code-block:: python

            from langchain.llms import OpenAI
            from langchain_experimental.chain_of_verification import CoVeChain
            llm = OpenAI(temperature=0)
            cove_chain = CoVeChain.from_llm(llm)
    """

    cove_chain: SequentialChain

    llm: Optional[BaseLanguageModel] = None
    baseline_response_prompt: PromptTemplate = BASELINE_RESPONSE_PROMPT
    plan_verifications_prompt: PromptTemplate = PLAN_VERIFICATIONS_PROMPT
    plan_verifications_output_model: BaseModel = PlanVerificationsOutputModel
    execute_verifications_prompt: PromptTemplate = EXECUTE_VERIFICATIONS_PROMPT
    revised_response_prompt: PromptTemplate = REVISED_RESPONSE_PROMPT
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
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        query = inputs[self.input_key]

        output = self.cove_chain.invoke(
            {"query": query}, callbacks=_run_manager.get_child()
        )
        return {self.output_key: output["result"]}

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

        baseline_response = baseline_response_prompt | llm | StrOutputParser()
        plan_verifcations_response: PlanVerificationsOutputModel = (
            {"baseline_response": baseline_response}
            | plan_verifications_prompt
            | llm
            | plan_verifications_output_parser
        )
        verification_questions = list(
            plan_verifcations_response.facts_and_verification_questions.values()
        )
        verify_chain = (
            {"verification_question": itemgetter("verification_question")}
            | execute_verifications_prompt
            | llm
            | StrOutputParser()
        )

        verify_results_str = ""
        for i in range(len(verification_questions)):
            question = verification_questions[i]
            answer = verify_chain.invoke({"verification_question": question})
            answer = answer.lstrip("\n")
            verify_results_str += f"Question: {question}\nAnswer: {answer}\n\n"

        map_ = RunnableMap(query=RunnablePassthrough())
        revise_response_chain = (
            {
                "query": map_,
                "baseline_response": baseline_response,
                "verified_responses": verify_results_str,
            }
            | revised_response_prompt
            | llm
            | StrOutputParser()
        )

        cove_chain = revise_response_chain
        return cls(
            cove_chain=cove_chain,
            **kwargs,
        )
