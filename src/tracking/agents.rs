use std::collections::HashMap;
use std::path::PathBuf;

/// Represents a single agent (main session or sub-agent).
#[derive(Debug, Clone)]
pub struct AgentNode {
    pub id: String,
    pub parent_id: Option<String>,
    pub session_file: PathBuf,
    pub label: String,
}

/// Tracks the hierarchy of agents in a session.
#[derive(Debug, Clone)]
pub struct AgentTree {
    pub agents: HashMap<String, AgentNode>,
    pub root_id: Option<String>,
}

impl AgentTree {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            root_id: None,
        }
    }

    pub fn add_agent(&mut self, agent: AgentNode) {
        if self.root_id.is_none() && agent.parent_id.is_none() {
            self.root_id = Some(agent.id.clone());
        }
        self.agents.insert(agent.id.clone(), agent);
    }

    pub fn children_of(&self, agent_id: &str) -> Vec<&AgentNode> {
        self.agents
            .values()
            .filter(|a| a.parent_id.as_deref() == Some(agent_id))
            .collect()
    }
}

impl Default for AgentTree {
    fn default() -> Self {
        Self::new()
    }
}
