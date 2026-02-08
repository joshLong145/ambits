use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{bail, eyre, Result};
use serde_pickle::value::{HashableValue, Value};
use sha2::{Digest, Sha256};

use crate::symbols::merkle::compute_merkle_hash;
use crate::symbols::{FileSymbols, ProjectTree, SymbolCategory, SymbolNode};

/// Scan a project using Serena's cached symbol data (.pkl files).
pub fn scan_project_serena(project_root: &Path) -> Result<ProjectTree> {
    let pkl_files = find_serena_caches(project_root);
    if pkl_files.is_empty() {
        bail!(
            "No Serena cache found at {}/.serena/cache/",
            project_root.display()
        );
    }

    let mut all_files = Vec::new();
    for pkl_path in &pkl_files {
        let data = fs::read(pkl_path)?;
        let value = serde_pickle::value_from_slice(&data, Default::default())
            .map_err(|e| eyre!("Failed to parse pickle {}: {}", pkl_path.display(), e))?;

        let is_raw = pkl_path
            .file_name()
            .map(|n| n == "raw_document_symbols.pkl")
            .unwrap_or(false);

        let files = if is_raw {
            parse_raw_pickle(&value)?
        } else {
            parse_document_pickle(&value)?
        };
        all_files.extend(files);
    }

    all_files.sort_by(|a, b| a.file_path.cmp(&b.file_path));

    Ok(ProjectTree {
        root: project_root.to_path_buf(),
        files: all_files,
    })
}

/// Find all Serena cache pickle files for a project.
/// Prefers raw_document_symbols.pkl over document_symbols.pkl per language.
pub fn find_serena_caches(project_root: &Path) -> Vec<PathBuf> {
    let cache_dir = project_root.join(".serena").join("cache");
    let entries = match fs::read_dir(&cache_dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut results = Vec::new();
    for entry in entries.flatten() {
        let lang_dir = entry.path();
        if !lang_dir.is_dir() {
            continue;
        }
        let raw = lang_dir.join("raw_document_symbols.pkl");
        if raw.exists() {
            results.push(raw);
            continue;
        }
        let doc = lang_dir.join("document_symbols.pkl");
        if doc.exists() {
            results.push(doc);
        }
    }
    results
}

/// Parse raw_document_symbols.pkl format.
/// Structure: {"__cache_version": (1,1), "obj": {path: (hash, [symbols])}}
fn parse_raw_pickle(value: &Value) -> Result<Vec<FileSymbols>> {
    let obj = dict_get(value, "obj").ok_or_else(|| eyre!("Missing 'obj' key in pickle"))?;
    let entries = as_dict(obj).ok_or_else(|| eyre!("'obj' is not a dict"))?;

    let mut files = Vec::new();
    for (key, val) in entries {
        let file_path_str = hashable_as_str(key).ok_or_else(|| eyre!("File key not a string"))?;
        let file_path = PathBuf::from(file_path_str);

        // Value is a tuple: (content_hash_str, [symbol_dicts])
        let items = as_tuple(val).ok_or_else(|| eyre!("File entry not a tuple"))?;
        if items.len() < 2 {
            continue;
        }
        let symbol_list =
            as_list(&items[1]).ok_or_else(|| eyre!("Symbol list not an array for {file_path_str}"))?;

        let path_prefix = file_path.to_string_lossy();
        let mut symbols = Vec::new();
        for sym_val in symbol_list {
            if let Ok(node) = convert_symbol(sym_val, &file_path, &path_prefix, "") {
                symbols.push(node);
            }
        }

        let total_lines = estimate_total_lines(&symbols);
        files.push(FileSymbols {
            file_path,
            symbols,
            total_lines,
        });
    }
    Ok(files)
}

/// Parse document_symbols.pkl format.
/// Structure: {"__cache_version": 3, "obj": {path: (hash, DocumentSymbols_state)}}
/// serde-pickle extracts the class instance as its __getstate__ dict.
fn parse_document_pickle(value: &Value) -> Result<Vec<FileSymbols>> {
    let obj = dict_get(value, "obj").ok_or_else(|| eyre!("Missing 'obj' key in pickle"))?;
    let entries = as_dict(obj).ok_or_else(|| eyre!("'obj' is not a dict"))?;

    let mut files = Vec::new();
    for (key, val) in entries {
        let file_path_str = hashable_as_str(key).ok_or_else(|| eyre!("File key not a string"))?;
        let file_path = PathBuf::from(file_path_str);

        let items = as_tuple(val).ok_or_else(|| eyre!("File entry not a tuple"))?;
        if items.len() < 2 {
            continue;
        }

        // The second element is the DocumentSymbols state dict
        let state = &items[1];
        let symbol_list = dict_get(state, "root_symbols")
            .and_then(as_list)
            .or_else(|| as_list(state)) // fallback: might be a plain list
            .ok_or_else(|| eyre!("Cannot find symbols for {file_path_str}"))?;

        let path_prefix = file_path.to_string_lossy();
        let mut symbols = Vec::new();
        for sym_val in symbol_list {
            if let Ok(node) = convert_symbol(sym_val, &file_path, &path_prefix, "") {
                symbols.push(node);
            }
        }

        let total_lines = estimate_total_lines(&symbols);
        files.push(FileSymbols {
            file_path,
            symbols,
            total_lines,
        });
    }
    Ok(files)
}

/// Convert a pickle Value dict into a SymbolNode.
fn convert_symbol(
    val: &Value,
    file_path: &Path,
    path_prefix: &str,
    parent_id: &str,
) -> Result<SymbolNode> {
    let name = dict_get(val, "name")
        .and_then(as_str)
        .ok_or_else(|| eyre!("Symbol missing 'name'"))?
        .to_string();

    let kind_int = dict_get(val, "kind")
        .and_then(as_i64)
        .unwrap_or(12); // default to Function
    let (category, label) = lsp_kind_to_category_and_label(kind_int);

    let (start_line, start_char, end_line, end_char) = extract_range(val);

    let id = if parent_id.is_empty() {
        format!("{path_prefix}::{name}")
    } else {
        format!("{parent_id}/{name}")
    };

    let line_count = if end_line > start_line {
        end_line - start_line + 1
    } else {
        1
    };

    // Content hash from identity (no source text available in raw format)
    let content_hash = {
        let mut hasher = Sha256::new();
        hasher.update(name.as_bytes());
        hasher.update(kind_int.to_le_bytes());
        hasher.update(start_line.to_le_bytes());
        hasher.update(end_line.to_le_bytes());
        hasher.finalize().into()
    };

    // Convert children
    let mut children = Vec::new();
    if let Some(child_list) = dict_get(val, "children").and_then(as_list) {
        for child_val in child_list {
            if let Ok(child) = convert_symbol(child_val, file_path, path_prefix, &id) {
                children.push(child);
            }
        }
    }

    let mut node = SymbolNode {
        id,
        name,
        category,
        label,
        file_path: file_path.to_path_buf(),
        byte_range: (start_line * 40 + start_char)..(end_line * 40 + end_char),
        line_range: (start_line + 1)..(end_line + 1), // 1-indexed like tree-sitter
        content_hash,
        merkle_hash: [0u8; 32],
        children,
        estimated_tokens: line_count * 15,
    };
    compute_merkle_hash(&mut node);
    Ok(node)
}

/// Extract range from a symbol dict: (start_line, start_char, end_line, end_char)
fn extract_range(val: &Value) -> (usize, usize, usize, usize) {
    let range = dict_get(val, "range");
    let start = range.and_then(|r| dict_get(r, "start"));
    let end = range.and_then(|r| dict_get(r, "end"));

    let start_line = start.and_then(|s| dict_get(s, "line")).and_then(as_i64).unwrap_or(0) as usize;
    let start_char = start
        .and_then(|s| dict_get(s, "character"))
        .and_then(as_i64)
        .unwrap_or(0) as usize;
    let end_line = end.and_then(|e| dict_get(e, "line")).and_then(as_i64).unwrap_or(0) as usize;
    let end_char = end
        .and_then(|e| dict_get(e, "character"))
        .and_then(as_i64)
        .unwrap_or(0) as usize;

    (start_line, start_char, end_line, end_char)
}

fn estimate_total_lines(symbols: &[SymbolNode]) -> usize {
    symbols
        .iter()
        .map(|s| s.line_range.end)
        .max()
        .unwrap_or(0)
}

fn lsp_kind_to_category_and_label(kind: i64) -> (SymbolCategory, String) {
    match kind {
        2 => (SymbolCategory::Module, "module".to_string()),
        3 => (SymbolCategory::Module, "namespace".to_string()),
        5 => (SymbolCategory::Type, "class".to_string()),
        6 => (SymbolCategory::Function, "method".to_string()),
        7 => (SymbolCategory::Variable, "property".to_string()),
        8 => (SymbolCategory::Variable, "field".to_string()),
        9 => (SymbolCategory::Function, "constructor".to_string()),
        10 => (SymbolCategory::Type, "enum".to_string()),
        11 => (SymbolCategory::Type, "interface".to_string()),
        12 => (SymbolCategory::Function, "function".to_string()),
        13 => (SymbolCategory::Variable, "variable".to_string()),
        14 => (SymbolCategory::Variable, "constant".to_string()),
        19 => (SymbolCategory::Implementation, "impl".to_string()),
        22 => (SymbolCategory::Variable, "enum_member".to_string()),
        23 => (SymbolCategory::Type, "struct".to_string()),
        26 => (SymbolCategory::Type, "type_param".to_string()),
        _ => (SymbolCategory::Unknown, "unknown".to_string()),
    }
}

// --- Value extraction helpers ---

fn dict_get<'a>(val: &'a Value, key: &str) -> Option<&'a Value> {
    match val {
        Value::Dict(entries) => {
            let target = HashableValue::String(key.to_string());
            for (k, v) in entries {
                if *k == target {
                    return Some(v);
                }
            }
            None
        }
        _ => None,
    }
}

fn as_str(val: &Value) -> Option<&str> {
    match val {
        Value::String(s) => Some(s),
        _ => None,
    }
}

fn hashable_as_str(val: &HashableValue) -> Option<&str> {
    match val {
        HashableValue::String(s) => Some(s),
        _ => None,
    }
}

fn as_i64(val: &Value) -> Option<i64> {
    match val {
        Value::I64(n) => Some(*n),
        Value::Int(n) => n.try_into().ok(),
        _ => None,
    }
}

fn as_list(val: &Value) -> Option<&[Value]> {
    match val {
        Value::List(l) => Some(l),
        _ => None,
    }
}

fn as_tuple(val: &Value) -> Option<&[Value]> {
    match val {
        Value::Tuple(t) => Some(t),
        _ => None,
    }
}

fn as_dict(val: &Value) -> Option<&BTreeMap<HashableValue, Value>> {
    match val {
        Value::Dict(d) => Some(d),
        _ => None,
    }
}
