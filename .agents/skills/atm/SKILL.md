---
name: atm
description: >-
  ATM (Amati, Tiru, Modifikasi) design and implementation framework.
  Use this skill whenever the user asks to perform an ATM workflow, replicate or benchmark existing reference code/outputs/dashboards/docs, or create solutions that match functional requirements while using uniquely modified code styles, phrasing, and PNG visual designs.
---

# ATM (Amati, Tiru, Modifikasi) Framework

The **ATM (Amati, Tiru, Modifikasi)** skill guides the agent to observe a reference input or goal, replicate its functional correctness, and modify its implementation so that the output is uniquely styled every time it is invoked.

---

## 🔴 MANDATORY FIRST STEP — Pre-Invocation Questions

**Before doing ANY work**, the agent MUST ask the user the following two questions interactively:

### 1. Username / Prefix
> "What is your **username or full name**? This will be used as the prefix on all output files that require one (e.g. dashboard PNGs named `{username}_dicoding-dashboard.png`, model files, or any submission artifact that requires a personal identifier)."

- Store the answer as `{USERNAME}`.
- Apply `{USERNAME}` as the prefix wherever a file name or artifact label calls for a personal identifier.
- Example: `William_dicoding-dashboard.png`, `william_notebook.ipynb`, etc.

### 2. Styling Preference
> "What **visual styling** would you like for the generated assets (dashboard PNG, charts, visuals)? You can pick one of the suggestions below or describe your own theme:"
>
> - 🌸 **Pastel Dreams** — Soft lavender, mint, powder blue, peach
> - 🌑 **Modern Dark Mode / Glassmorphism** — Deep navy with neon highlights
> - ⬜ **Sleek Monochrome** — Minimalist grayscale with a single accent color
> - 🍂 **Warm Earthy** — Terracotta, beige, sage green, warm ivory
> - 🌌 **Retro Synthwave / Cyberpunk** — Dark violet, neon magenta, cyan, bright yellow
> - ❄️ **Nordic Clean Light** — Crisp white/ice-blue, deep indigo, teal, slate grey
> - 🎨 **Custom** — Describe your own palette or mood board

- Store the chosen style as `{STYLE_THEME}`.
- Apply `{STYLE_THEME}` throughout **all** visual assets generated in this ATM run.
- If the user says "surprise me" or "anything", dynamically pick one of the above themes — but **never default to the same one twice in a row**.

> [!IMPORTANT]
> Do NOT proceed to AMATI → TIRU → MODIFIKASI until both `{USERNAME}` and `{STYLE_THEME}` have been confirmed.

---

## The 3 Core Pillars

```
+-------------------+      +-------------------+      +-----------------------+
|  1. AMATI         |  --> |  2. TIRU          |  --> |  3. MODIFIKASI        |
|  (Observe Spec)   |      |  (Match Output)   |      |  (Unique Styling)     |
+-------------------+      +-------------------+      +-----------------------+
```

### 1. AMATI (Observe & Analyze)
- Inspect the reference specification, dataset, existing implementation, or user prompt requirements thoroughly.
- Identify all mandatory output criteria, key performance indicators (KPIs), expected file formats, required parameters, and functional deliverables.
- Never guess data schemas or criteria without inspecting the source context.

### 2. TIRU (Imitate Functional Target)
- Ensure 100% functional match with the desired output criteria.
- Match exact data calculations, metric definitions, pipeline execution logic, and requirement coverage.
- Verify that tests pass, data outputs line up, and submission criteria are completely satisfied.

### 3. MODIFIKASI (Modify & Differentiate)
Every time this skill is invoked, dynamically vary the execution so the output feels custom and non-generic.
Use `{USERNAME}` and `{STYLE_THEME}` collected in the Pre-Invocation Questions above.

#### A. File Naming & Prefix
- All output assets that require a personal identifier MUST be prefixed with `{USERNAME}`.
  - Dashboard PNG: `{USERNAME}_dicoding-dashboard.png`
  - Notebook (if applicable): `{USERNAME}_notebook.ipynb`
  - Any submission artifact requiring a user label follows the same pattern.

#### B. Wording & Natural Phrasing (Bahasa / English)
- Avoid canned text, cookie-cutter templates, or identical explanations.
- Vary title phrasing, markdown headings, summary sections, and narrative commentary across iterations.
- Change variable names, docstrings, function descriptions, and commit message wording while preserving meaning.

#### C. Code Style & Architecture
- Alternate programming paradigms or refactoring styles (e.g., modular functions vs. class-based pipelines, list comprehensions vs. explicit loops, custom helper abstractions).
- Use distinct variable naming conventions (e.g., snake_case aliases, descriptive domain-specific names).
- Organize imports, functions, and modular script layouts cleanly but uniquely.

#### D. Visual Design & Graphic Aesthetics (PNG / Dashboards / Visuals)
Whenever generating visual assets (e.g. Matplotlib, Seaborn, PIL graphics, or Web UIs), apply the `{STYLE_THEME}` chosen at invocation time:

- **Color Scheme**: Use the palette from `{STYLE_THEME}` as the primary design language for all charts, KPI cards, backgrounds, and text colors.
- **Layout & Typography Variations**: Re-arrange KPI card placement, font typography (bold headers, custom padding), grid structure (2×3 vs 3×2, split vertical vs horizontal header), and chart styles (horizontal bar vs vertical, donut vs pie, violin vs boxplot).
- **Unique Design Signature**: Add custom card borders, subtle gradient accents, or custom icon/legend styling so no two generated images look identical.

---

## Execution Workflow

1. **Pre-Invocation**: Ask for `{USERNAME}` and `{STYLE_THEME}`. Confirm before proceeding.
2. **Observe Phase**: Read requirements and document the target specification.
3. **Replicate Phase**: Build the core logic to guarantee exact output correctness.
4. **Differentiate Phase**: Apply the **Modifikasi** rules above — prefix files with `{USERNAME}`, apply `{STYLE_THEME}` to all visuals, vary wording and code structure.
5. **Validation & Name Matching Phase**:
   - Run unit tests / execution checks to ensure functional accuracy.
   - **Consolidated Name Check**: Before zipping the files for submission, inspect every text file (e.g. `README.md`, `notebook.ipynb`, code scripts, and configuration metadata) inside the submission directory.
   - Ensure *every single occurrence* of the developer name or placeholder matches the `{USERNAME}` exactly. No other dummy names or mismatched names must be present in comments, cells, or markdown headings. Correct any discrepancies.
   - Package/zip the files only after this verification passes.

