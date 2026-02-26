---
paths:
  - "smcjax/**/*.py"
  - "tests/**/*.py"
---

# Design & Code Principles

Apply these principles when writing, reviewing, or refactoring code.

## KISS — Keep It Simple, Stupid

Favor the simplest solution that works. Complexity is a liability.

- If two approaches solve the problem, pick the one with fewer moving
  parts.
- Avoid clever tricks that trade readability for brevity.
- A solution you can explain in one sentence is almost always better
  than one that needs a paragraph.

## YAGNI — You Aren't Gonna Need It

Don't build features or abstractions until you actually need them.

- Write code for today's requirements, not tomorrow's hypotheticals.
- Delete speculative code paths, unused parameters, and "just in case"
  logic.
- If a need arises later, add it then — with a test first (TDD).

## SOLID

The five principles apply beyond strict OOP — they guide module and
service design too.

### Single Responsibility (SRP)

A module, class, or function should have one reason to change.

- If a function does two things, split it into two functions.
- If a module mixes I/O with business logic, separate them.

### Open/Closed (OCP)

Software entities should be open for extension but closed for
modification.

- Use protocols, callbacks, or strategy patterns to allow new behavior
  without editing existing code.
- Prefer adding new functions/classes over modifying existing ones when
  extending behavior.

### Liskov Substitution (LSP)

Subtypes must be substitutable for their base types without breaking
correctness.

- If you override a method, honor the parent's contract (same
  preconditions, same or stronger postconditions).
- Prefer composition when substitution semantics get awkward.

### Interface Segregation (ISP)

No client should be forced to depend on methods it does not use.

- Keep protocols and ABCs small and focused.
- Prefer multiple narrow protocols over one wide one.

### Dependency Inversion (DIP)

Depend on abstractions, not concretions.

- High-level modules should not import low-level modules directly;
  both should depend on protocols or interfaces.
- Use dependency injection to pass collaborators rather than
  hard-coding them.

## Composition over Inheritance

Prefer composing behavior from small, focused units rather than deep
inheritance hierarchies.

- Use functions, protocols, and dataclasses to compose behavior.
- Limit inheritance depth to two levels. If you need more, refactor
  to composition.
- Mixins are acceptable when they add a single, orthogonal behavior.

## Separation of Concerns

Each module, layer, or service should own one well-defined
responsibility.

- Keep I/O at the edges; keep core logic pure and testable.
- Don't mix configuration parsing, business rules, and persistence
  in the same function.
- Organize code so that each file or module answers one question:
  "What does this do?"

## Principle of Least Astonishment

APIs and interfaces should behave the way a reasonable user would
expect.

- Function names should describe what they do — no hidden side
  effects.
- Default arguments should be safe and unsurprising.
- Error messages should tell the caller what went wrong and what to
  do about it.
- Follow Python community conventions (PEP 8 naming, context
  managers for resources, iterators for sequences).
