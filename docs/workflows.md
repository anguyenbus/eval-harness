# Why Workflows Matter for the Case Assistant Project

**Context:** Australian taxation law. The case assistant supports case officers across income tax, GST, FBT, superannuation, transfer pricing, R&D Tax Incentive, and anti-avoidance work — reviewing taxpayer submissions, objections, private ruling applications, audit responses, and related correspondence.

This report explains what a workflow is, why workflows are worth investing in, and provides seven fully drafted examples grounded in everyday tax case work. Every architectural claim has been verified against a reference implementation; the few that warrant qualification are flagged transparently in Section 8.

---

## 1. The Opportunity

Case officers already do excellent extraction work — they read carefully, structure their findings, and apply judgement honed over years. The challenge isn't capability; it's *reach*. A senior officer's careful analysis of a Part IVA matter today helps the file they're working on, and not much else. The same officer reviewing thirty objections in a batch will, by hour six, want their hour-one rigour back.

Workflows offer a simple promise: take the analytical patterns that already exist in your team's heads and turn them into reusable, shareable templates. The officer's expertise stays theirs; the workflow just lets them apply it at scale, share it with colleagues, and improve it over time.

## 2. What Workflows Are

A workflow is a saved, named, reusable extraction template. It has:

- a stable prompt (and, optionally, a structured column schema),
- a practice-area tag (Income Tax, GST, Transfer Pricing, and so on),
- sharing controls so a team can collaborate on a single source of truth.

Two flavours are supported:

- **Assistant workflows** apply a single prompt to a single document and produce a narrative output — well-suited to summaries, structured reviews, and the kind of analysis an officer would write up in a file note.
- **Tabular workflows** apply a column schema to many documents and produce a comparable grid. Each column has its own prompt and an output format (`text`, `bulleted_list`, `date`, `monetary_amount`, `yes_no`, `tag`, `number`, `percentage`, `currency`). The grid is ideal for triage, batch review, and any task where comparing across files matters.

Each output cell is also tagged with a traffic-light flag — **green** (standard or favourable), **yellow** (needs attention), **red** (material concern), **grey** (not addressed or neutral) — so an officer scanning a grid can immediately see where to focus.

## 3. What Workflows Do for the Team

**Capture and share expertise.** A senior officer's framing of "what counts as a scheme for Part IVA" lives in a template that the whole team can apply. The expertise compounds rather than walking out the door at the next staff movement.

**Consistency across files.** Officer A and Officer B running the same workflow ask the same questions in the same order. That makes batch comparison meaningful — a team leader can scan thirty objection grids and trust that "amount in dispute" means the same thing in every row.

**Speed without shortcuts.** A workflow doesn't change what gets reviewed; it changes how quickly the officer can get to the parts that need their judgement. Selecting a template takes seconds; the saved time goes into the questions that actually require human consideration.

**Practice-area focus.** Workflows are tagged and filtered by practice area, so an officer working a GST file sees the GST templates first and isn't wading through transfer pricing options.

**Collaboration.** Workflows are shared by email (or by team), with an `allow_edit` flag that lets a designated group refine the prompt centrally. When the senior tax counsel sharpens the Part IVA prompt, every officer with the workflow gets the improvement on their next run.

**Composability.** Built-in workflows ship with the platform; custom workflows are created by the team. Officers can hide built-ins they don't use, and build on the ones that work for their practice. Nothing is forced on anyone.

## 4. Performance Architecture

For tabular reviews, the reference implementation processes every selected document concurrently. Within each document, all columns are handed to a single streaming LLM call that emits one structured JSON result per column as it works through them, with the grid filling in column-by-column in the UI rather than waiting for a sequential pass.

The headline often quoted — "a 20-column review in roughly a second" — is illustrative rather than benchmarked, and the precise wall time will depend on the model, the document length, and network conditions. The substantive point is the architecture: one batched streaming call per document, not twenty sequential round-trips. For an officer running an objection-triage workflow across thirty taxpayer files, the difference is the workflow finishing while they make a coffee, rather than running while they go to lunch.

---

## 5. Seven Worked Examples

Each example below is presented as it would appear in the workflow library: a title, a practice tag, an indication of type, and the actual prompt content. Where the workflow is tabular, the column schema is given in full with the prompt for each column. Where it is an assistant workflow, the complete `prompt_md` is shown.

All examples reference Australian tax legislation, ATO practice, and the kinds of documents a case officer reviews in the ordinary course of work.

### Example 1 — Objection Triage

- **Type:** Tabular
- **Practice:** Income Tax
- **Use case:** An officer receives a batch of objection cases under Part IVC of the *Taxation Administration Act 1953* and needs to sort them by complexity, ground, and risk before allocation.

**Columns:**

| # | Name | Format | Prompt |
|---|---|---|---|
| 0 | Taxpayer | `text` | Identify the taxpayer making the objection. State the full legal name, TFN or ABN if shown, and entity type (individual, company, trust, partnership, or SMSF). |
| 1 | Income years in dispute | `bulleted_list` | List every income year or tax period referred to in the objection. One year per bullet, in chronological order. |
| 2 | Amount in dispute | `monetary_amount` | State the total amount in dispute across all years. Aggregate primary tax, shortfall interest charge, general interest charge, and any administrative penalties. If the figure is not stated, state "Not Found". |
| 3 | Grounds of objection | `bulleted_list` | List each ground of objection separately, in the taxpayer's own framing. One ground per bullet. Do not paraphrase into ATO terminology — preserve how the taxpayer has put it. |
| 4 | Statutory provisions cited | `bulleted_list` | List every statutory provision the taxpayer relies on. Use full citations (e.g. "section 8-1 of the ITAA 1997", "Subdivision 815-B of the ITAA 1997", "section 165-5 of the GST Act"). |
| 5 | Cases cited | `bulleted_list` | List every case authority the taxpayer relies on, with full citation including court and year (e.g. "Commissioner of Taxation v Spotless Services Ltd (1996) 186 CLR 404"). |
| 6 | Lodged in time? | `yes_no` | Was the objection lodged within the period prescribed by section 14ZW of the TAA 1953? If the objection is lodged out of time and is accompanied by a request under section 14ZW(2) for the Commissioner to deal with it as if it were lodged within time (to be considered by the Commissioner under section 14ZX), answer "No" and note the extension request in the reasoning. |
| 7 | Evidence attached | `bulleted_list` | List every document the taxpayer has attached or referred to as supporting their objection (e.g. contracts, bank statements, valuations, expert reports, statutory declarations). |
| 8 | Complexity | `tag` (tags: `Routine`, `Specialist`, `Complex`) | Assess complexity. Use "Routine" for straightforward factual or computational matters with settled law; "Specialist" where specialist input (transfer pricing, international, financial services) is likely needed; "Complex" where the matter is novel, precedential, or involves multiple intersecting issues. |

**Result:** One row per objection. A team leader scans thirty objections and sees immediately which are routine, which need specialist eyes, and which warrant a complex-matter pathway — before a single officer has been allocated.

### Example 2 — Private Ruling Application Summary

- **Type:** Assistant
- **Practice:** Income Tax / GST
- **Use case:** An officer reviewing a private ruling application needs a structured summary to attach to the case file before drafting the Commissioner's response.

**Prompt:**

```markdown
## Private Ruling Application — Structured Summary

Review the attached private ruling application and produce a structured summary covering the matters below. For each section, identify the key facts or contentions, quote the relevant parts of the application with paragraph references, and flag any matters that need clarification from the applicant before the Commissioner can be satisfied of the scheme.

1. **Applicant** — Full legal name, TFN or ABN, entity type, and tax agent if applicable.

2. **Scheme or arrangement** — A neutral description of the scheme on which the ruling is sought, in your own words. Note where the applicant's description appears incomplete, inconsistent, or framed in conclusory terms.

3. **Questions for ruling** — Each question the applicant has asked the Commissioner to rule on, set out separately and verbatim.

4. **Period of ruling** — The income years or tax periods to which the ruling is to apply.

5. **Applicant's contentions** — For each question, summarise the applicant's argument and the reasoning chain they have offered.

6. **Statutory provisions engaged** — Every provision the application invokes or that appears engaged on the facts (e.g. specific Divisions and sections of the ITAA 1936, ITAA 1997, TAA 1953, GST Act, FBT Act).

7. **Authorities cited** — Every case, public ruling (TR, TD, GSTR, MT, etc.), and ATO guidance product the applicant relies on.

8. **Anti-avoidance considerations** — Identify any aspects that may engage Part IVA, the GAAR in Division 165 of the GST Act, the multinational anti-avoidance law in section 177DA, or the diverted profits tax in section 177J. Note these even if the applicant has not raised them.

9. **Factual gaps** — Matters of fact that are stated in conclusory terms, not supported by documents, or omitted entirely, where the Commissioner would need further information before being satisfied of the scheme as described.

10. **Preliminary risk assessment** — A short paragraph identifying the matters most likely to require careful consideration or specialist input.

Deliver the summary inline in the chat response. Do not generate a Word document unless asked.
```

**Result:** A consistent file-note structure for every private ruling application that comes in. Junior officers produce the same level of structured analysis as senior officers; senior officers spend their time on the matters the summary has flagged, not on re-creating the summary itself.

### Example 3 — Part IVA Red-Flag Scan

- **Type:** Tabular
- **Practice:** Anti-Avoidance
- **Use case:** Across a portfolio of restructure documents or scheme disclosures, identify which arrangements look like they have the dominant purpose of obtaining a tax benefit.

**Columns:**

| # | Name | Format | Prompt |
|---|---|---|---|
| 0 | Scheme description | `text` | Describe the arrangement in neutral terms, using the steps as set out in the documents. Do not adopt the taxpayer's labels (e.g. "commercial restructure") if those labels would beg the question. |
| 1 | Identified tax benefit | `bulleted_list` | Identify each tax benefit obtained or sought, in the sense of section 177C. Include deductions, capital losses, franking credits, withholding reductions, deferrals, and reductions in assessable income. |
| 2 | Counterfactual | `text` | State the alternative postulate — what would reasonably have happened absent the scheme, on the evidence available in the documents. If multiple counterfactuals are plausible, list them and identify which the documents most strongly support. |
| 3 | Manner of entry | `text` | Section 177D(2)(a): the manner in which the scheme was entered into or carried out. Identify any features that go beyond what an ordinary commercial transaction would require. |
| 4 | Form and substance | `text` | Section 177D(2)(b): the form and substance of the scheme. Identify any mismatch between legal form and economic substance. |
| 5 | Timing | `text` | Section 177D(2)(c): the time at which the scheme was entered into and the length of the period during which it was carried out. Note any timing that aligns suspiciously with year-end or with the realisation of an offsetting gain or loss. |
| 6 | Tax result | `text` | Section 177D(2)(d): the result, in relation to the operation of the Act, that the scheme produces. Quantify if possible. |
| 7 | Financial position change | `text` | Section 177D(2)(e): any change in the financial position of the relevant taxpayer that has resulted, will result, or may reasonably be expected to result, from the scheme. |
| 8 | Connected party position change | `text` | Section 177D(2)(f): any change in the financial position of any person connected with the relevant taxpayer. |
| 9 | Other consequences | `text` | Section 177D(2)(g): any other consequence for the relevant taxpayer or connected person. |
| 10 | Nature of connection | `text` | Section 177D(2)(h): the nature of the connection between the relevant taxpayer and any connected person. |
| 11 | Commercial substance | `bulleted_list` | List indicators of independent commercial substance: third-party negotiation, market-rate pricing, independent advice, pre-existing commercial relationship, ordinary commercial timing. |
| 12 | Artificiality indicators | `bulleted_list` | List indicators of artificiality: circular flows of funds, back-to-back arrangements, steps without independent function, special-purpose entities introduced solely for the scheme, contrived timing. |
| 13 | Dominant purpose — preliminary view | `tag` (tags: `Likely`, `Arguable`, `Unlikely`) | A preliminary view, weighing the eight matters as a whole, on whether a person who entered into or carried out the scheme would reasonably be concluded to have had the dominant purpose of enabling the relevant taxpayer to obtain a tax benefit. This is a triage signal, not a determination. |

**Result:** Across a batch of restructure files, the officer sees at a glance which matters warrant a full Part IVA analysis and which are within ordinary commercial dealing. The eight-matter framework is applied consistently to every file — which is exactly what an eventual determination would need to demonstrate.

### Example 4 — Transfer Pricing Local File Review

- **Type:** Tabular
- **Practice:** Transfer Pricing
- **Use case:** Review a taxpayer's local file (or Subdivision 815-D documentation) against the substantive requirements of Subdivision 815-B and the ATO's documentation expectations.

**Columns:**

| # | Name | Format | Prompt |
|---|---|---|---|
| 0 | Tested party | `text` | Identify the tested party for each international related party dealing covered by the file. State the entity name and the rationale given for choosing it as the tested party. |
| 1 | Related party dealings | `bulleted_list` | List every category of international related party dealing covered (e.g. inbound services, outbound services, royalties, interest on intra-group financing, tangible goods, cost contribution arrangements). |
| 2 | Functional analysis — functions | `bulleted_list` | List the functions performed by each party to the dealings as identified in the file. |
| 3 | Functional analysis — assets | `bulleted_list` | List the assets used by each party, distinguishing routine assets from non-routine intangibles. |
| 4 | Functional analysis — risks | `bulleted_list` | List the risks assumed by each party, and note whether the file analyses control of risk and financial capacity to bear risk consistent with the OECD Transfer Pricing Guidelines. |
| 5 | Method selected | `tag` (tags: `CUP`, `RPM`, `CPM`, `TNMM`, `PSM`, `Other`) | Identify the transfer pricing method selected for each dealing. |
| 6 | Method justification | `text` | Summarise the justification offered for the method selected. Note any failure to address why other methods were rejected. |
| 7 | Comparables — source | `text` | Identify the source of the comparables (e.g. commercial database, internal comparables) and the date of the search. |
| 8 | Comparables — selection criteria | `bulleted_list` | List the quantitative and qualitative selection criteria applied to the comparable set, including geographic, industry, size, and independence filters. |
| 9 | Arm's length range | `text` | State the arm's length range derived (interquartile or full range) and the financial indicator used (e.g. operating margin, return on assets, Berry ratio). |
| 10 | Outcomes testing | `yes_no` | Does the file show that the taxpayer's actual outcome for the income year falls within the arm's length range? |
| 11 | Reconciliation | `yes_no` | Does the file reconcile the tested transactions to the audited financial statements? |
| 12 | STPR eligibility considered | `yes_no` | Does the file address whether the taxpayer is eligible for any Simplified Transfer Pricing Record Keeping (STPR) option under PCG 2017/2? |
| 13 | CbC linkage | `yes_no` | Does the file refer to and reconcile with the Country-by-Country reporting submissions where the taxpayer is a significant global entity? |
| 14 | Material gaps | `bulleted_list` | List any matters required by Subdivision 815-D, the ATO's local file requirements, or the OECD Transfer Pricing Guidelines that are missing, inadequately addressed, or unsupported by evidence in the file. |

**Result:** A standardised diagnostic across the taxpayer's documentation, with material gaps surfaced explicitly. The officer's judgement is still required on whether the gaps matter — the workflow just ensures none are missed.

### Example 5 — GST Ruling Request Categorisation

- **Type:** Assistant
- **Practice:** GST
- **Use case:** Quickly determine what kind of GST issue a private ruling request actually raises before assigning to a specialist.

**Prompt:**

```markdown
## GST Ruling Request — Issue Categorisation

Review the attached GST private ruling application and produce a categorisation note covering the matters below.

1. **Primary issue category** — Identify which of the following the request principally concerns:
   - Supply classification (taxable, GST-free, input taxed)
   - Attribution and timing
   - Input tax credit entitlement and apportionment
   - Going concern exemption (section 38-325 of the GST Act)
   - Margin scheme (Division 75)
   - Financial supplies and reduced input tax credits (Division 70)
   - Cross-border supplies and the connected with Australia tests (Division 9, Subdivision 38-E)
   - Adjustments, refunds, and credit notes (Divisions 19, 21, 142)
   - Grouping, joint ventures, and related entities (Divisions 48, 51)
   - Other (specify)

2. **Statutory provisions engaged** — List every GST Act provision invoked or engaged on the facts, with full citations.

3. **Public guidance to review** — List every relevant ATO ruling (GSTR series), determination (GSTD), and practice statement that the officer should consider before responding.

4. **Specialist area** — Indicate whether the matter falls within general GST, real property, financial services, cross-border, or another specialist area.

5. **Facts that drive classification** — Identify the two to four facts in the application that are doing the heaviest lifting in the analysis, and any facts that appear missing or stated only in conclusory terms.

6. **Preliminary view** — Offer a tentative view on the question asked, identifying the key uncertainty, if any. This is a triage signal for case allocation, not the Commissioner's view.

Deliver the categorisation inline.
```

**Result:** Routing decisions get made in minutes rather than after the first specialist has already started reading.

### Example 6 — Division 7A Loan Agreement Check

- **Type:** Tabular
- **Practice:** Income Tax — Private Groups
- **Use case:** Triage a batch of complying loan agreements supplied by a private group during a review, to identify which warrant closer examination.

**Columns:**

| # | Name | Format | Prompt |
|---|---|---|---|
| 0 | Lender | `text` | Identify the lender. State the full legal name and confirm it is a private company for Division 7A purposes. |
| 1 | Borrower | `text` | Identify the borrower. State the full legal name and the relationship to the lender (shareholder, associate, or entity interposed). |
| 2 | Principal amount | `monetary_amount` | State the principal amount of the loan. |
| 3 | Date of loan | `date` | State the date the loan was made. |
| 4 | Agreement in place by lodgement day? | `yes_no` | Was a written loan agreement put in place before the private company's lodgement day for the income year in which the loan was made, as required by section 109N of the ITAA 1936? |
| 5 | Term | `text` | State the term of the loan. Confirm whether it is consistent with the maximum permitted term — seven years for an unsecured loan, or twenty-five years for a loan secured by a registered mortgage over real property under section 109N(3). |
| 6 | Interest rate | `percentage` | State the interest rate. |
| 7 | Rate meets benchmark? | `yes_no` | Does the interest rate meet or exceed the Division 7A benchmark rate for the relevant income year? |
| 8 | Minimum yearly repayment status | `tag` (tags: `Met`, `Shortfall`, `Not Yet Due`, `Unclear`) | Based on the documents provided, has the minimum yearly repayment under section 109E been met for each completed income year since the loan was made? |
| 9 | Security | `text` | If the loan is treated as a twenty-five year loan, describe the security and confirm whether it meets the requirements of section 109N(3)(b) including the 110% market value test. |
| 10 | Refinancing or sub-trust indicators | `bulleted_list` | List any indicators that the loan refinances an earlier loan, replaces a UPE, or relates to a sub-trust arrangement that requires its own Division 7A analysis. |

**Result:** Across a private group with many intra-group loans, the officer sees in a single grid which loans are compliant, which have repayment shortfalls, and which have structural issues that warrant a deeper look.

### Example 7 — Audit Response Letter Summary

- **Type:** Assistant
- **Practice:** General
- **Use case:** The taxpayer has responded to a position paper. The officer needs a structured comparison of what the ATO put to the taxpayer versus what the taxpayer has now said in reply.

**Prompt:**

```markdown
## Audit Response — Position-Paper Comparison

The attached document is the taxpayer's response to an ATO position paper. Produce a structured comparison covering the matters below. Where the position paper itself is attached or has been provided, work from both documents; where only the response is provided, work from the response alone and note that the position paper was not provided.

For each issue identified in the position paper (or, if the position paper is not provided, each issue addressed in the response):

1. **Issue** — A short label for the issue (e.g. "Deductibility of management fees", "Section 25-90 interest deductions", "Apportionment of input tax credits").

2. **ATO position (as put)** — A concise statement of the ATO's position on the issue, with paragraph references to the position paper if available.

3. **Taxpayer response** — A concise statement of the taxpayer's response, with paragraph references to the response.

4. **New facts asserted** — Any facts asserted in the response that were not before the case officer when the position paper was drafted. Identify whether documentary evidence has been supplied to support the new facts.

5. **New authorities relied on** — Any cases, rulings, or ATO guidance products relied on in the response that were not addressed in the position paper.

6. **Concessions** — Matters on which the taxpayer has accepted, in whole or in part, the ATO's position.

7. **Silences** — Matters from the position paper that the response does not address.

8. **Factual disputes outstanding** — Where the response asserts facts contrary to those in the position paper and the question turns on disputed primary fact, identify the dispute and the kinds of further information that might resolve it (including, where appropriate, a formal information request under section 353-10 of Schedule 1 to the TAA 1953).

9. **Suggested next step** — A short recommendation for each issue: accept the taxpayer's response, maintain the position, issue a further information request, or escalate for specialist input.

Deliver the comparison inline as a structured note the officer can attach to the file.
```

**Result:** A consistent way of receiving audit responses across the team. The structure makes it harder for important concessions or silences to be missed, and easier for a team leader reviewing the file to see what has actually changed.

---

## 6. The Practical Difference

| Without Workflows | With Workflows |
| --- | --- |
| Each officer writes prompts from scratch each file | Selecting a template takes seconds; saved time goes into judgement |
| Different officers ask different questions on the same document type | Standardised extraction across the team |
| A junior officer's first batch looks different from a senior officer's | A junior using the same workflow starts with the senior's structure |
| Strong analytical patterns live in one person's chat history | Strong patterns live in a shared library, available to everyone |
| Workflow gains don't compound across the team | Each improvement to a shared workflow benefits everyone on their next run |

The minute-level estimates often associated with this kind of tooling — for example, "thirty minutes per file becomes thirty seconds" — are reasonable rules of thumb rather than measurements. The structural reason they hold is straightforward: a workflow turns prompt-crafting into prompt-selecting, with the column schema or prompt already written, reviewed, and refined over time.

## 7. Suggested Path Forward

These are options for the team to consider, not prescriptions:

**Start with the highest-volume document types.** Objections, private ruling applications, audit responses, and Division 7A loan reviews recur constantly and benefit most from standardisation. A workflow library that covers the team's top half-dozen document types delivers most of the value.

**Make workflow authorship a recognised contribution.** When an officer crafts a strong prompt for a recurring matter, supporting them to convert it into a shared workflow — with a senior officer's review — captures the value of that work for everyone. Treating workflow contributions as recognised work, not unbilled overhead, is what builds the library over time.

**Govern the shared library lightly but deliberately.** Decide who can edit shared workflows, how prompt changes are reviewed, and how built-ins are extended rather than re-implemented. The library is the operationalised form of the team's analytical method — a small amount of stewardship keeps it coherent.

**Treat the library as an asset.** The accumulated workflows represent codified expertise. They are worth backing up, versioning, and considering when planning succession and onboarding.

## 8. Verification Summary

| Claim | Status |
| --- | --- |
| Workflows have a standardised prompt and (optionally) a structured column schema | Verified in the reference implementation |
| Practice-area tagging and filtering | Verified |
| Sharing by email with `allow_edit` flag | Verified |
| Hide what you don't use | Verified |
| Built-in vs. custom workflows coexist | Verified |
| Document-level parallelism in tabular reviews | Verified |
| Per-document column streaming via a single batched LLM call | Verified |
| Output formats listed (`text`, `bulleted_list`, `date`, `monetary_amount`, `yes_no`, `tag`, `number`, `percentage`, `currency`) | Verified — these are the formats the reference implementation supports |
| Traffic-light cell flags (green / yellow / red / grey) | Verified |
| "20 columns in ~1 second" headline figure | **Illustrative, not benchmarked.** The architecture is one streaming call per document, so the real number depends on the model, document length, and network conditions. The qualitative claim (much faster than sequential round-trips) is sound. |
| "30 min → 30 sec per file" ROI | Plausible estimate, not measured in the reference implementation |
| Australian tax law citations in Section 5 | Verified against current Commonwealth legislation and ATO published guidance (Part IVC and section 14ZW of the TAA 1953; section 175A of the ITAA 1936; subsection 177D(2) of the ITAA 1936; sections 177DA and 177J of the ITAA 1936; Subdivisions 815-B and 815-D of the ITAA 1997; PCG 2017/2; section 109N of the ITAA 1936; section 38-325, Divisions 70, 75, 165 and Subdivision 38-E of the GST Act; Divisions 19, 21 and 142 of the GST Act; section 353-10 of Schedule 1 to the TAA 1953). |

The worked examples in Section 5 use Australian taxation law concepts (Part IVC and section 14ZW of the TAA 1953, Part IVA and the eight matters in subsection 177D(2) of the ITAA 1936, Subdivisions 815-B and 815-D of the ITAA 1997, Division 7A and section 109N of the ITAA 1936, various GST Act provisions, section 353-10 of Schedule 1 to the TAA 1953, and PCG 2017/2). Statutory references are drawn from current Commonwealth tax legislation; officers should still verify any specific section number against the current consolidation before publishing a workflow to the live library.