"""Import exactly the bytes read under the run's existing file authority."""
from __future__ import annotations

import base64
from pathlib import Path

from .workspace import ToolPolicyError, scoped_workspace


async def import_document(args, context):
    if context.knowledge is None or not context.conversation_id:
        raise ToolPolicyError("Document import requires a conversation knowledge base")
    context.check_cancel()
    with scoped_workspace(context, args["path"]) as (workspace, relative, target):
        raw, _ = workspace.read(relative, 128 * 1024 * 1024)
        source = Path(target or context.workspace / relative).absolute().as_uri()
    # Pass the pinned bytes, never reopen an unchecked host path after approval.
    document = await context.knowledge.attach(context.conversation_id, filename=Path(args["path"]).name,
        content_base64=base64.b64encode(raw).decode(), allow_visuals=context.allow_images,
        source_origin={"source": source, "operation_id": context.operation_id, "run_id": context.run_id})
    context.check_cancel()
    return {"document_id": document["id"], "source_sha256": document["sha256"],
            "source": source, "operation_id": context.operation_id, "run_id": context.run_id,
            "name": document["name"], "chunks": document["chunks"],
            "instructions_trusted": False, "next": "Use search_knowledge or read_knowledge for source passages"}
