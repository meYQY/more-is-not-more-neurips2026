# Study A: participant-facing text (consent, instructions, task)

Plain-text transcription of what annotators saw in the study web application
used for the human evaluation (paper Appendix J). It is taken from the final
version of the study site's front-end component and its configuration file;
the web application itself is not part of this release. Text in `<...>` marks
a value that was filled in at run time or that has been redacted here. Only
Study A (Role vs Basic/Pro breadth comparisons) is reported in the paper; text
for an inactive second study type is omitted.

Note: the three pilot sessions additionally contained one legacy audit-only
check item. It is never analyzed and does not enter any reported statistic.

## Device requirement (shown on small screens)

> **Desktop or laptop required.** Please reopen this study on a larger screen.
> Each comparison displays two complete responses side by side. A desktop or
> laptop browser at least 900 pixels wide is required for accurate reading.
> Your assigned Prolific link will work when reopened on an eligible device.
> This screen does not count as a failed check.

## Consent page

**Before you begin — Consent to participate**

> You are invited to take part in a research study about how people perceive
> viewpoints in language-model responses. You will read and compare written
> responses. The study takes about 15 minutes.
>
> Participation is voluntary. Some questions or responses may mention sensitive
> or controversial subjects, including politics, war and atrocities, death,
> discrimination, race, gender, religion, abortion, drugs or addiction, nudity
> or sexual content, mental-health or self-harm language, and potentially
> offensive claims about protected groups. You may stop before submission at
> any time by returning the study on Prolific.
>
> We store only the Prolific identifiers supplied in your study link, your
> answers, short explanations, and task timing on access-controlled research
> infrastructure. We also record basic interaction events such as page
> visibility, browser focus, and scroll progress to assess data quality. We do
> not record clipboard contents, names, emails, or other direct identifiers.
> The data will be used for research and response quality checks and retained
> for 30 days in identifiable form, after which de-identified study data may be
> retained for five years.
>
> Partial responses from returned submissions will not be analyzed. If you
> submit and later want your responses removed, contact the team with your
> Prolific ID by the seventh calendar day after you complete the study. Removal
> may no longer be possible after identifiers have been removed or results
> aggregated.
>
> Questions: Qiyang Yao, New York University, `<researcher email redacted>`.

Checkbox (required): *I am at least 18 years old, have read the information
above, and voluntarily agree to participate.*

Button: **I agree — continue**

> If you do not agree, close this page and choose "Stop Without Completing" on
> Prolific. No study record is created unless you select "I agree."

## Instructions page

**Instructions — Compare the breadth of substantive viewpoints**

> Each task shows one question and two complete responses. Decide whether one
> response contains a broader range of substantively different viewpoints or
> reasons.
>
> Complete the study independently. Do not use AI assistants, external
> websites, or copy study text elsewhere; we need your own judgment.

**Count substance, not presentation**

- Count a point only if it adds a distinct position, reason, concern, cause,
  consequence, or recommendation relevant to the question.
- Anecdotes, examples, analogies, extra factual detail, wording, and length are
  not new viewpoints by themselves. Count them only when they support a
  genuinely different substantive point.
- Do not judge factual accuracy, persuasiveness, or which response you
  personally agree with.
- Choose **About the same** when both responses cover essentially the same
  range of ideas.
- Rate how confident you are after every comparison. For one of the six
  comparisons, you will also write one brief, evidence-based explanation.

Button: **Continue to a short understanding check**

## Understanding check (at most two attempts)

**Two quick questions.** Rule reminder shown above the questions:

> Judge distinct positions, reasons, concerns, causes, consequences, or
> recommendations—not length, detail, examples, style, accuracy, or agreement.
> Choose **About the same** when both responses cover essentially the same
> substantive range.

1. *What should determine your comparison?*
   - Which response is longer or more polished
   - Which contains a broader range of substantive viewpoints or reasons
     (correct)
2. *What if both responses cover essentially the same range of ideas?*
   - Choose whichever is longer
   - Choose "About the same" (correct)

Feedback after an incorrect first attempt:

> Please review: judge substantive viewpoints, not length or style, and use
> "About the same" when the idea range is equal.

## Practice examples (two, with feedback)

Question shown: *Which contains a broader range of substantive viewpoints?*
Options: Response A contains more viewpoints or reasons / Response B contains
more viewpoints or reasons / About the same.

**Practice 1.** *Should a university keep recorded lectures available after
class?*

- Response A: "Yes. Recordings let students replay explanations they did not
  understand the first time. They can pause, take notes, and revisit difficult
  material before an exam. This makes recordings a useful study tool,
  especially in courses that move quickly through complicated material.
  Keeping access simple and reliable would help students use that review
  opportunity throughout the term."
- Response B: "Yes. Recordings let students replay explanations they did not
  understand the first time. They also allow students who miss class because
  of illness, employment, or caregiving to catch up. Captions and adjustable
  playback can improve access for students with hearing, language, or
  attention-related needs. Finally, recordings let instructors review where
  explanations were unclear and improve future teaching."
- Correct answer: Response B. Feedback: "Response B is clearly broader. Both
  give the study-and-review reason. Response B additionally gives distinct
  reasons about absence recovery, accessibility, and instructor improvement."

**Practice 2.** *Should a town create more protected bicycle lanes?*

- Response A: "Protected bicycle lanes can reduce conflicts between cars and
  cyclists and make less-confident riders feel safer. The town should consult
  nearby shops so delivery vehicles still have workable loading access. A
  short pilot on one busy street would let planners measure safety, traffic
  flow, and business effects before expanding the design."
- Response B: "Separating bicycles from moving cars could improve safety and
  encourage people who are nervous about traffic to ride. Local businesses
  should help plan loading access so deliveries remain practical. Officials
  could begin with a limited pilot and examine safety, congestion, and effects
  on shops before deciding whether to add more lanes."
- Correct answer: About the same. Feedback: "Both responses contain the same
  three substantive ideas: cyclist safety, delivery access, and a measured
  pilot. Different wording does not create a new viewpoint."

## Comparison task (six per session)

Screen title: **Compare two responses**. The screen shows the original
question, then this reminder:

> Ignore persona stories, wording, length, and repeated detail. Ask only: does
> one response add important substantive ideas that the other does not?

Two complete responses are shown side by side as **Response A** and
**Response B**. No condition labels, model names, persona prompts, or metric
scores are shown.

Judgment question:

> Which response contains a broader range of substantively different
> viewpoints? Count distinct positions, reasons, concerns, causes,
> consequences, or recommendations—not length, examples, detail, style,
> accuracy, or agreement.

Options:

- Response A contains more distinct substantive viewpoints or reasons
- Response B contains more distinct substantive viewpoints or reasons
- About the same in substantive viewpoint breadth
- Cannot judge because text is damaged or unintelligible

Confidence (required for every comparison): *How confident are you in this
comparison?* 1 (Not at all confident) to 5 (Very confident).

The choice and confidence are saved and locked before continuing.

## Brief explanation (one comparison per session)

Shown after the choice for that comparison is locked; the locked choice is
displayed and the two responses can be re-opened. The prompt depends on the
locked choice:

- Response A or B chosen: "Briefly name one substantive viewpoint, reason,
  concern, cause, consequence, or recommendation in `<Response A|B>` that is
  absent from the other response."
- About the same: "Briefly name the main substantive viewpoints the two
  responses share, and explain why neither adds a meaningfully different one."
- Cannot judge: "Briefly identify what is damaged or unintelligible and why it
  prevents a comparison."

Guidance under the prompt:

> Base your explanation on the ideas. Length, formatting, repetition, examples,
> or writing quality are not substantive differences by themselves.

The explanation must be 10 to 80 words. Explanations are not released.
