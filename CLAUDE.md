# CLAUDE.md

A self-graded course: blank notebooks in `notebooks/` are **generated** from `src/authoring/<module>/<NN_name>/build.py`. The grading for each notebook lives in `src/selfgrade/<module>/<name>.py`.

- To write, extend, review or quiz, use the `/tutor` skill in `.claude/skills/tutor/`. Its `style.md` and `format.md` are mandatory reading before editing any notebook.
- Never edit `.ipynb` files in `notebooks/` by hand. Edit the builder, then run `python src/authoring/verify.py [mXX/NN]`, which must end with **ALL OK**.
- Never edit `workspace/`: it holds the learner's own solutions. The only exceptions are `python -m selfgrade update|reset`, or an explicit request from the learner.
- Every change is proposed and approved before it's applied, and is made on a new branch, never directly on `main` (see the skill's Change workflow).
- `curriculum.yaml` is the map of what is covered where. Keep it in sync with the notebooks.
