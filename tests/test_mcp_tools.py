import asyncio

from atlandex_agent.mcp_tools import McpToolBox, _as_text


class SlowTool:
    name = "search_video"

    async def ainvoke(self, args):
        await asyncio.sleep(5)
        return "never"


class OkTool:
    name = "find_videos"

    async def ainvoke(self, args):
        return [{"type": "text", "text": '{"ok": true}'}]


class BoomTool:
    name = "locate_quote"

    async def ainvoke(self, args):
        raise ValueError("bad phrase")


def run(coro):
    return asyncio.run(coro)


def test_a_stalled_tool_call_returns_an_error_text_instead_of_hanging():
    tb = McpToolBox([SlowTool()], call_timeout_sec=0.05)
    assert "no response after" in run(tb.call("search_video", {}))


def test_content_blocks_are_flattened_to_text():
    tb = McpToolBox([OkTool()])
    assert run(tb.call("find_videos", {})) == '{"ok": true}'
    assert _as_text(("plain", None)) == "plain"


def test_unknown_tool_and_tool_exceptions_become_error_text_for_the_model():
    tb = McpToolBox([BoomTool()])
    assert run(tb.call("nope", {})).startswith("Error: unknown tool")
    assert "bad phrase" in run(tb.call("locate_quote", {}))
