use anyhow::Result;
use std::io::{self, Write};

/// Ask the user to confirm the plan that was just printed, returning whether
/// the operation should proceed.
///
/// With `skip` set (the `--no-plan` flag) the prompt is bypassed and the
/// operation proceeds immediately. Pressing Enter (empty answer) proceeds; a
/// closed/empty stdin (e.g. piped input or CI without `--no-plan`) declines
/// rather than hang.
pub fn confirm(skip: bool) -> Result<bool> {
    if skip {
        return Ok(true);
    }
    print!("\nProceed with these steps? [Y/n] ");
    io::stdout().flush()?;
    let mut input = String::new();
    if io::stdin().read_line(&mut input)? == 0 {
        // EOF: no interactive terminal. Don't proceed silently; the caller
        // should pass --no-plan for non-interactive runs.
        println!();
        return Ok(false);
    }
    let answer = input.trim().to_ascii_lowercase();
    Ok(answer.is_empty() || answer == "y" || answer == "yes")
}
