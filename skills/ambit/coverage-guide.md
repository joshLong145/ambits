# Coverage Interpretation Guide

## Reading the Report Strategically

### For Architecture Understanding
- Files with high Seen% but low Full% indicate Claude saw the structure (classes, function signatures) but didn't examine implementations
- This is common for files where only the API surface matters
- If you need Claude to understand *how* something works (not just *what* it does), ensure Full% is high

### For Code Review Confidence
- Critical files (core business logic, security-sensitive code) should have high Full%
- Utility/helper files may acceptably have lower Full% if their interfaces are clear
- Test files with low coverage may indicate untested assumptions

### For Debugging Sessions
- Low coverage on files related to a bug suggests Claude may be missing context
- Ask Claude to read specific files to increase coverage before making recommendations

## Identifying Knowledge Gaps

### Red Flags
```
src/auth/token_validator.rs        20.0%    5.0%   <- Security-critical, needs attention
src/core/payment_processor.rs      15.0%    0.0%   <- Core logic barely seen
src/api/handlers/mod.rs           100.0%   10.0%   <- Saw exports only, not implementations
```

### Healthy Patterns
```
src/main.rs                       100.0%   80.0%   <- Entry point well understood
src/lib.rs                        100.0%   60.0%   <- Public API examined
src/config.rs                      90.0%   90.0%   <- Configuration fully read
```

## Coverage by File Type

### Source Files (*.rs, *.py, *.ts, *.java)
- Aim for 70%+ Full% on files you're modifying
- 50%+ Seen% across the module gives structural awareness

### Test Files (*_test.rs, test_*.py)
- High coverage means Claude understands expected behavior
- Useful for understanding invariants and edge cases

### Configuration Files
- Often have 100% Full% since they're typically read completely
- Low coverage here may mean missing environment/setup context

## Using Coverage to Guide Conversations

### Before Refactoring
```bash
ambits -p . --coverage
# Check that target files have Full% > 60%
# If not, ask Claude to read them first
```

### Before Architectural Decisions
```bash
ambits -p . --coverage
# Verify Seen% > 80% across the module/package
# Ensures Claude has structural awareness
```

### When Claude's Suggestions Seem Off
```bash
ambits -p . --coverage
# Look for low coverage on relevant files
# Claude may be missing important context
```

## Symbol-Level Depth (Interactive TUI)

The TUI shows coverage at the symbol level with color coding:
- **Green**: Full body read - complete understanding
- **Yellow**: Partially seen - signature/overview only
- **Red/Dim**: Unseen - no visibility

Navigate the tree to identify exactly which functions or classes need more attention.
