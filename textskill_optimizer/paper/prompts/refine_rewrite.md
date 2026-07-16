You are refining a proposed set of full-skill revision suggestions.

Use the same trajectory minibatch and the prior suggestions to improve precision,
generality, and consistency. Do not modify or target the protected slow-update field.
Return `converged=true` only when another refinement would not materially improve
the suggestions.

Respond ONLY with a valid JSON object:
{
  "reasoning": "<brief semantic refinement>",
  "revise_suggestions": [
    {
      "type": "add_rule|remove_rule|merge_rules|reorganize|compress|clarify",
      "title": "<short title>",
      "motivation": "<why this matters>",
      "instruction": "<what the rewriting optimizer should change>",
      "priority_hint": "high|medium|low"
    }
  ],
  "converged": <boolean>
}
