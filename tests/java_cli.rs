//! Process-level test for Java: a `.java` file gets a symbol tree through the
//! binary, and `show` resolves a method by id.

mod common;

use common::run_ambits;

const SHAPES: &str = r#"package com.example;

import java.util.List;

/** Shapes and the things that draw them. */
public class Shapes {
    private final List<String> names;

    public Shapes(List<String> names) {
        this.names = names;
    }

    public void draw() {
        draw(1);
    }

    public void draw(int scale) {
        names.forEach(n -> new Point(scale, scale));
    }

    interface Renderer {
        default boolean visible() {
            return true;
        }
    }

    enum Kind {
        CIRCLE, SQUARE;

        Kind next() {
            return values()[(ordinal() + 1) % values().length];
        }
    }

    record Point(int x, int y) {}
}
"#;

#[test]
fn a_java_file_dumps_as_a_tree_and_its_methods_show_by_id() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/Shapes.java"), SHAPES).unwrap();

    let dump = run_ambits(root, &["--dump", "--full"]);
    let tree: Vec<&str> = dump
        .lines()
        .skip_while(|l| !l.contains("src/Shapes.java"))
        .skip(1)
        .map(|l| l.split(" [").next().unwrap())
        .collect();
    assert_eq!(
        tree,
        vec![
            "    class Shapes",
            "      field names",
            "      constructor Shapes",
            "      method draw",
            "      method draw",
            "      interface Renderer",
            "        method visible",
            "      enum Kind",
            "        method next",
            "      record Point",
        ],
        "{dump}"
    );

    let shown: serde_json::Value =
        serde_json::from_str(&run_ambits(root, &["show", "src/Shapes.java::Shapes/Renderer/visible"])).unwrap();
    let only = &shown["results"][0]["matches"][0];
    assert_eq!(only["label"], "method");
    assert_eq!(only["lines"], serde_json::json!([22, 24]));
    assert!(only["definition"].as_str().unwrap().starts_with("default boolean visible()"));

    let overloads: serde_json::Value =
        serde_json::from_str(&run_ambits(root, &["show", "src/Shapes.java::Shapes/draw"])).unwrap();
    assert_eq!(overloads["results"][0]["matches"].as_array().unwrap().len(), 2, "both overloads match the id");
}
