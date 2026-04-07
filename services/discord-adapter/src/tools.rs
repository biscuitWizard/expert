use expert_types::signals::ToolDefinition;

/// Returns the full set of Discord domain tool definitions.
/// All have `is_domain_tool: true` so the llm-gateway routes them
/// to `actions.{stream_id}` for execution by this adapter.
pub fn discord_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "send_message".to_string(),
            description: "Send a message to a Discord channel.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "The ID of the channel to send the message to"
                    },
                    "content": {
                        "type": "string",
                        "description": "The message content to send"
                    }
                },
                "required": ["channel_id", "content"]
            }),
            is_domain_tool: true,
        },
        ToolDefinition {
            name: "reply_to_message".to_string(),
            description: "Reply to a specific message in a Discord channel.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "The ID of the channel containing the message"
                    },
                    "message_id": {
                        "type": "string",
                        "description": "The ID of the message to reply to"
                    },
                    "content": {
                        "type": "string",
                        "description": "The reply content"
                    }
                },
                "required": ["channel_id", "message_id", "content"]
            }),
            is_domain_tool: true,
        },
        ToolDefinition {
            name: "send_dm".to_string(),
            description: "Send a direct message to a Discord user.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the user to DM"
                    },
                    "content": {
                        "type": "string",
                        "description": "The message content to send"
                    }
                },
                "required": ["user_id", "content"]
            }),
            is_domain_tool: true,
        },
        ToolDefinition {
            name: "react_to_message".to_string(),
            description: "Add a reaction emoji to a message.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "The ID of the channel containing the message"
                    },
                    "message_id": {
                        "type": "string",
                        "description": "The ID of the message to react to"
                    },
                    "emoji": {
                        "type": "string",
                        "description": "The emoji to react with (unicode emoji or custom format name:id)"
                    }
                },
                "required": ["channel_id", "message_id", "emoji"]
            }),
            is_domain_tool: true,
        },
        ToolDefinition {
            name: "join_guild".to_string(),
            description: "Join a Discord server using an invite code or link.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "invite_code": {
                        "type": "string",
                        "description": "The invite code or full invite URL (e.g. 'abc123' or 'https://discord.gg/abc123')"
                    }
                },
                "required": ["invite_code"]
            }),
            is_domain_tool: true,
        },
        ToolDefinition {
            name: "leave_guild".to_string(),
            description: "Leave a Discord server.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "guild_id": {
                        "type": "string",
                        "description": "The ID of the guild to leave"
                    }
                },
                "required": ["guild_id"]
            }),
            is_domain_tool: true,
        },
        ToolDefinition {
            name: "send_friend_request".to_string(),
            description: "Send a friend request to a Discord user by username.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "The username of the person to send a friend request to"
                    }
                },
                "required": ["username"]
            }),
            is_domain_tool: true,
        },
        ToolDefinition {
            name: "accept_friend_request".to_string(),
            description: "Accept a pending incoming friend request.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the user whose friend request to accept"
                    }
                },
                "required": ["user_id"]
            }),
            is_domain_tool: true,
        },
        ToolDefinition {
            name: "remove_friend".to_string(),
            description: "Remove a user from your friends list.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the friend to remove"
                    }
                },
                "required": ["user_id"]
            }),
            is_domain_tool: true,
        },
        ToolDefinition {
            name: "typing_indicator".to_string(),
            description: "Show a typing indicator in a channel (lasts ~10 seconds).".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "The ID of the channel to show typing in"
                    }
                },
                "required": ["channel_id"]
            }),
            is_domain_tool: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_count() {
        let defs = discord_tool_definitions();
        assert_eq!(defs.len(), 10);
    }

    #[test]
    fn all_are_domain_tools() {
        for def in &discord_tool_definitions() {
            assert!(def.is_domain_tool, "{} should be a domain tool", def.name);
        }
    }

    #[test]
    fn expected_tool_names() {
        let defs = discord_tool_definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"send_message"));
        assert!(names.contains(&"reply_to_message"));
        assert!(names.contains(&"send_dm"));
        assert!(names.contains(&"react_to_message"));
        assert!(names.contains(&"join_guild"));
        assert!(names.contains(&"leave_guild"));
        assert!(names.contains(&"send_friend_request"));
        assert!(names.contains(&"accept_friend_request"));
        assert!(names.contains(&"remove_friend"));
        assert!(names.contains(&"typing_indicator"));
    }

    #[test]
    fn schemas_have_required_fields() {
        for def in &discord_tool_definitions() {
            let schema = &def.parameters_schema;
            assert_eq!(
                schema["type"], "object",
                "{} schema should be an object",
                def.name
            );
            assert!(
                schema.get("properties").is_some(),
                "{} schema missing properties",
                def.name
            );
            assert!(
                schema.get("required").is_some(),
                "{} schema missing required",
                def.name
            );
        }
    }

    #[test]
    fn names_are_unique() {
        let defs = discord_tool_definitions();
        let mut names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        let original_len = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), original_len, "tool names must be unique");
    }

    #[test]
    fn descriptions_non_empty() {
        for def in &discord_tool_definitions() {
            assert!(
                !def.description.is_empty(),
                "{} has empty description",
                def.name
            );
        }
    }
}
