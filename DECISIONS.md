# DECISIONS.md

## Overview

The primary objective of BriteSpark was to create a policy assistant capable of answering questions using the relevant excerpts from the provided policy documents rather than the model's training data.

I wanted the system to do more than simply retrieve text and pass it to Gemini, the answer should be tied to specific pieces of evidence, dates should be handled correctly, citations to supporting paragraphs should be checked, and it should refuse to answer when there is insufficient evidence.

The system had to be simple enough to demonstrate and test.

---

## 1. Policy Parser

The first step was to parse the policy documents into a structured format.

Each parsed clause contains information such as:

- a clause ID

- its title

- the text of the clause

- source document

- its amendments, when applicable

I opted to implement this design so that the remainder of the application has access to the relevant details for each policy clause.

The original policy manual is treated as the source of truth and is not altered in any way.

---

## 2. Policy Retriever

The second step was to implement retrieval.

I chose to implement a deterministic BM25-style retriever that can accept a query and return relevant policy clauses.

I ruled out the option of implementing a vector database for this purpose since the corpus of information is relatively small, and such a solution would bloat the codebase and add operational complexity for little benefit.

The retriever's deterministic nature also makes testing easier, which is critical for this application.

I wanted to ensure that the answer is generated using relevant evidence rather than allowing Gemini to perform a free-form search.

---

## 3. Basic Grounded Answer Generation

With the retriever in place, Part 3 of the application was straightforward.

The following components were used to generate grounded answers:

User question → Policy retriever → Relevant policy clauses → Gemini → Grounded answer

Gemini was instructed to use the relevant policy clauses to formulate its answer.

At this stage, I only aimed to demonstrate a working grounded-answer pipeline.

I did not want Gemini to act as a policy database, so the application retrieves the relevant evidence and passes it to the model for processing.

---

## 4. Surprise Challenge: Temporal Policy Versioning

The Surprise Challenge required adjustments to Part 3 of the application.

The issue the challenge introduced was that the user's question could include multiple dates, and not all of them may be relevant to the query.

A policy amendment could take effect on one date while the determination is made on another date.

Using the first, the last, the earliest, or the latest date would produce incorrect results.

Rather than modifying the answer-generation pipeline, I decided to augment it with a temporal policy versioning layer that is implemented in `src/temporal.py` .

The temporal engine recognizes semantic date roles such as:

- `change_date`

- `determination_date`

- `claim_period_date`

- `span_start`

- `span_end`

Once these roles are identified, the temporal engine applies the relevant policy versioning logic to select the appropriate policy clause(s).

### Why I did not want Gemini to perform policy versioning

I did not want versioning decisions to be made by Gemini, as the task requires deterministic processing.

The temporal engine handles date roles by applying the amendment logic directly.

For example:

- Reporting deadlines are based on the change date under Amendment §5.2

- Earnings disregard is based on the determination date under Amendment §5.1

- Spanning periods are based on applicable start/end dates under §5.3 and §7.4.3

### An adversarial example

One of the examples that required temporal policy versioning to work correctly was:

"A change occurred on 20 February 2026 and the determination was made on 20 March 2026. What reporting deadline applies?"

The correct answer is the pre-amendment rule, as the change date of 20 February 2026 precedes 1 March 2026.

The later determination date of 20 March 2026 is irrelevant to the reporting deadline.

The reverse scenario was also implemented:

"A change occurred on 20 March 2026 and the determination was made on 20 February 2026. What reporting deadline applies?"

The change date of 20 March 2026 falls after 1 March 2026, and the post-amendment rule should be used.

This was the primary reason for implementing temporal policy versioning with explicit date roles rather than relying on Gemini to handle such scenarios.

### Temporal states

The temporal engine classifies the query into one of the following categories:

- `PRE_AMENDMENT`

- `POST_AMENDMENT`

- `SPANNING`

- `UNSPECIFIED`

This information is subsequently used by the answer-generation component.

### Testing

The surprise challenge was tested using queries that utilized:

- change and determination dates

- determination-date earnings disregard

- pre-amendment reporting

- post-amendment reporting

- a claim period with a later determination

- spanning periods

- unspecified dates

The policy manual itself was not altered.

---

## 5. Citations

I improved upon the initial design after addressing the Surprise Challenge.

Gemini should not be able to make up policy citations.

The application checks the citations included in an answer for validity by comparing them to the relevant evidence.

If a citation is not found in the relevant evidence, it will not be considered valid, and the answer will no longer be considered fully grounded.

I also added filters to prevent the application from mistakenly treating numbers or other text as policy citations.

For example, the following would not be classified as valid citations:

- "10 calendar days"

- "$175"

- "2026"

- "20 February 2026"

The system stores validated citations in a structured format that includes:

- a citation ID

- source document

- clause ID

- clause title

- clause text

- indication of whether a clause has been amended

- transitional information, if applicable

- provenance

This allows the system to relay this information to the user upon request.

---

## 6. Refusal / Insufficient Evidence

I implemented the "do not answer if there is no relevant policy" rule as a general design principle.

If there is no relevant policy evidence, the application will refuse to answer and provide an explanation rather than guessing.

For example, a question such as "What is the capital of France?" would be classified as having insufficient evidence since the topic is unrelated to the provided policies.

Even if Gemini were able to provide an answer, the system would be unable to ground it in any relevant policy evidence.

### Zero-evidence fast path

If the application detects that the retrieved policy evidence is completely irrelevant, it can refuse to answer immediately without consulting Gemini.

The refusal message always includes the following information:

- `grounded = False`

- `insufficient_evidence = True`

- no validated citations

- no unsupported citations

This helps prevent unnecessary processing.

### Partial evidence

I did not implement the design where the application refuses to answer to any evidence that it cannot validate.

If a part of the question can be answered using relevant policy evidence, the application will attempt to do so.

However, the part of the question that cannot be answered will also be explicitly noted.

This approach avoids both hallucinations and unconditional refusal to answer.

---

## 7. Conflict Handling

I ran into an issue while implementing the policy versioning logic that caused me to rethink the way the application handles conflicting evidence.

The issue was that the reporting rules in the original and amended policy both have different requirements.

However, the fact that the two provisions do not align does not make them conflicting.

I resolved this issue by decoupling policy versioning from policy conflict resolution.

The above scenario represents a temporal difference in the rules rather than a substantive disagreement.

The temporal engine is responsible for determining the applicable version of the rule, while the conflict resolver examines substantive disagreements between the retrieved pieces of evidence.

If two retrieved provisions contradict each other, the system can recognize this rather than arbitrarily selecting one of them.

When a conflict is detected, the application:

- identifies the relevant clauses

- keeps both as evidence

- asks Gemini to explain how the retrieved evidence differs

- cites both pieces of evidence

- does not attempt to invent a rule about which version takes precedence

- keeps the answer grounded by citing both pieces of evidence

This approach allows the application to resolve conflicts without introducing additional layers of complexity.

---

## 8. Evaluation and Testing

I wanted the evaluation suite to be deterministic and not rely on a few manually written test cases.

I implemented a test suite that contains 21 test cases.

The test suite covers:

- direct policy questions

- similarly worded questions

- pre-amendment questions

- post-amendment questions

- determination-date cases

- spanning periods

- unspecified dates

- unsupported questions

- refusal cases

- citation validation

- citation completeness

- conflict detection

- partial evidence

- adversarial mixed-date questions

- retrieval stability

- multi-clause questions

The current evaluation results are as follows:

- Evidence Retrieval Accuracy: 100%

- Temporal Classification Accuracy: 100%

- Citation Validity Rate: 95.2%

- Citation Completeness Rate: 95.2%

- Refusal Safety Accuracy: 100%

- Conflict Detection Accuracy: 100%

There are 109 individual tests in the final regression suite.

Two of the tests are currently skipped when the application does not have a Gemini API key.

---

## 9. UI

For the final UI, I opted for simplicity and used a conversational format rather than attempting to design a complex dashboard.

I removed any superfluous elements and opted to focus on relevant details.

The application opens in a centered and simple format.

After asking a question, the conversation view is opened, where the user's messages and the assistant's responses are displayed.

The UI keeps the user informed of relevant evidence and highlights such details as grounding status, evidence sufficiency, temporal status, conflict status, and other supporting information.

Rather than using a separate front-end framework, I used Flask, HTML, and CSS to implement a simple UI with minimal overhead.

I did not want to have to rely on an entire front-end framework when a simple solution would suffice.

---

## 10. What I Rejected

### Vector database

I rejected the idea of using a vector database for this project due to the complexity and operational overhead associated with it.

### Agent workflows

I did not pursue the option of building a large agent-based workflow.

### Letting Gemini decide what to cite

I did not rely on Gemini to decide what to cite.

I implemented evidence citation validation to ensure that Gemini cannot cite information that was not retrieved.

### Letting Gemini decide conflicting rules

I did not want Gemini to decide which rule takes precedence when rules conflict.

### General knowledge Q&A

I did not want the system to engage in general knowledge QA since it would undermine the primary purpose of the application.

The system is designed to adhere strictly to the policy documents; if the answer is not there, the system should refuse to answer rather than invent information.

### Front-end framework

I did not want to use a front-end framework such as React since the project would not benefit from such an involved solution.

---

## 11. What I Cut For Time

Time constraints prevented me from implementing certain features that I would have liked to include in the final application.

In particular, I did not implement:

- user accounts

- a production-grade database for storing conversations

- an admin dashboard

- document management

- document upload infrastructure

- production infrastructure

- vector similarity search

- analytics

- a separate front-end framework

- a policy management system

These features are plausible additions for a production-grade solution, but they had to be cut for this prototype.

---

## 12. What the System Is Not

The system does not guarantee that the underlying policy is correct.

The system does not:

- act as a replacement for a policy officer or a caseworker

- determine which policy takes precedence

- invent rules that are not present in the source material

- answer unsupported questions about general knowledge

- automatically resolve policy contradictions

- guarantee that every retrieved clause is the only possible answer

- replace human judgment for policy-related matters

The system serves to retrieve the available evidence, apply the deterministic rules I have written for the challenge, generate a grounded response, and expose this information to the user.

---

## 13. What I Would Do Next

If I had more time to work on this application, I would expand the evaluation suite.

I would add more test cases for:

- complex date roles

- similarly worded questions

- policy contradictions

- citation validation

- partial evidence

- multi-clause questions

I would improve the conflict detection engine so that it can recognize additional policy contradictions beyond simply having different values for the same field.

For a production-grade application, I would implement the following:

- user-facing authentication

- conversation storage

- analytics

- production-grade infrastructure

- policy versioning

- an administrative review system

- comprehensive testing

These additions would allow the application to evolve beyond a research prototype and into a full-fledged production service with long-term viability.

---

## 14. Final Design Principle

The single most important design principle that guided my decisions was the following:

Do not allow the system to suggest information that cannot be reliably supported by the provided policy documents.

I initially designed Part 3 to implement a basic retrieve-and-generate architecture.

However, the Surprise Challenge revealed that the system needed additional processing related to temporal policy versioning.

The ability to cite relevant evidence, detect conflicts, and implement deterministic citation validation were all built upon this foundation.

The final application is relatively small and deterministic, which is appropriate for the task at hand.