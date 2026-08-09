"""
Default tool specs the LLM can call. Whatever a persona lists in `tools_enabled`
is advertised to the model using these schemas; when the model calls one, the
orchestrator relays it to your configured tools.webhook_url and feeds the
JSON result back. The actual behaviour lives in YOUR webhook - this file only
describes the shapes.

Add your own by extending TOOL_SPECS (same shape) and listing the name in a
persona's tools_enabled.

Derived from ICTContact (https://www.ictcontact.com).
ICT Innovations (https://www.ictinnovations.com) - ICT Vision (https://ict.vision)
Author: Tahir Almas. MIT licensed.
"""

TOOL_SPECS = {
    "transfer": {
        "description": "Transfer the current call to a human agent, queue, or "
                       "extension. Use when the caller asks for a person or you "
                       "cannot help.",
        "parameters": {
            "target": {"type": "string",
                       "description": "Destination, e.g. 'queue:sales', "
                                      "'ext:1001', or a bare queue name."},
            "reason": {"type": "string",
                       "description": "Short reason for the transfer, for the agent."},
        },
    },
    "schedule_callback": {
        "description": "Schedule a callback to this caller at a later time.",
        "parameters": {
            "when": {"type": "string",
                     "description": "When to call back, ISO 8601 or natural language "
                                    "your webhook can parse (e.g. 'tomorrow 3pm')."},
            "note": {"type": "string", "description": "What the callback is about."},
        },
    },
    "mark_dnc": {
        "description": "Add this caller to the Do-Not-Call list. Use only when "
                       "the caller explicitly asks not to be contacted again.",
        "parameters": {
            "reason": {"type": "string", "description": "Why the caller was marked DNC."},
        },
    },
    "crm_lookup": {
        "description": "Look up the caller or an account in the CRM.",
        "parameters": {
            "query": {"type": "string",
                      "description": "Phone number, email, name, or account id to search."},
        },
    },
    "crm_update": {
        "description": "Update a field on the caller's CRM record.",
        "parameters": {
            "field": {"type": "string", "description": "Field name to update."},
            "value": {"type": "string", "description": "New value."},
        },
    },
    "send_sms": {
        "description": "Send an SMS to the caller (e.g. a link or confirmation).",
        "parameters": {
            "text": {"type": "string", "description": "Message body to send."},
        },
    },
}
