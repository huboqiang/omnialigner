import os
import re
from getpass import getpass
import nest_asyncio
import asyncio
import platform
import base64
import time
import io
from PIL import Image
import numpy as np
from typing import List, Optional
from typing_extensions import TypedDict
from datetime import datetime

from IPython import display
from dotenv import load_dotenv
from langchain_core.runnables import chain as chain_decorator
from langchain_core.messages import BaseMessage, SystemMessage
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_modelscope import ModelScopeChatEndpoint
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from playwright.async_api import async_playwright
from playwright.async_api import Page
from prompt import prompt_napari

timestamp_init = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = f"./out_{timestamp_init}"
os.makedirs(out_dir, exist_ok=True)

def base64_to_image(base64_str: str) -> np.ndarray:
    """
    Convert a base64 string to an image.
    """
    # Remove the prefix if it exists
    
    base64_decoded = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(base64_decoded))
    image_np = np.array(image)

    return image_np


class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str


class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]


# This represents the state of the agent
# as it proceeds through execution
class AgentState(TypedDict):
    page: Page  # The Playwright web page lets us interact with the web environment
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]  # The bounding boxes from the browser annotation function
    prediction: Prediction  # The Agent's output
    # A system message (or messages) containing the intermediate steps
    scratchpad: List[BaseMessage]
    observation: str  # The most recent response from a tool


async def click(state: AgentState):
    # - Click [Numerical_Label]
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = click_args[0]
    
    # Extract only the leading integer from bbox_id
    match = re.match(r"\d+", bbox_id)
    if not match:
        return f"Error: invalid bbox_id format: {bbox_id}"
    bbox_id_int = int(match.group())
    try:
        bbox = state["bboxes"][bbox_id_int]
    except Exception:
        return f"Error: no bbox for : {bbox_id_int}"
    x, y = bbox["x"], bbox["y"]
    if bbox["type"] == "roi":
        await page.evaluate(f"goToRegion({bbox['x']}, {bbox['y']}, {bbox['w']}, {bbox['h']}, zoom = 10, immediately = true)")
        return f"Clicked ROI {bbox_id}"
    
    
    await page.mouse.click(x, y)
    return f"Clicked {bbox_id}"

    


async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return (
            f"Failed to type in element from bounding box labeled as number {type_args}"
        )
    bbox_id = type_args[0]
    bbox_id = int(bbox_id)
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    await page.mouse.click(x, y)
    # Check if MacOS
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"


async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args

    if target.upper() == "WINDOW":
        # Not sure the best value for this:
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        # Scrolling within a specific element
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"


async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."


async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."


async def to_viewer(state: AgentState):
    page = state["page"]
    await page.goto("http://127.0.0.1:5020/viewer")
    return "Navigated to duckduckgo.com."

cmd_roi = """
const bp = $('#basePathInput').val();
$.getJSON(`/mark_bbox/${bp}`, data => {
    markBoxes(data.boxes);
});
"""

cmd_unmark = """
const bp = $('#basePathInput').val();
$.getJSON(`/unmark_bbox/${bp}`, () => {
    clearMarks();
});
"""

@chain_decorator
async def mark_page(page: Page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            bboxes_1 = await page.evaluate(cmd_roi)
            bboxes += [ {
                "x": bbox["x"],
                "y": bbox["y"],
                "w": bbox["w"],
                "h": bbox["h"],
                "type":"roi",
                "text": bbox["label"],
                "ariaLable":"" 
            } for bbox in bboxes_1["boxes"] ]
            break
        except Exception:
            time.sleep(3)

    screenshot = await page.screenshot()
    
    # Ensure the bboxes don't follow us around
    # await page.evaluate("unmarkPage()")
    await page.evaluate(cmd_unmark)
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }


async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])

    img1 = base64_to_image(marked_page["img"])
    # Image.fromarray(img1).save("out.png")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Image.fromarray(img1).save(f"{out_dir}/out_{timestamp}.png")
    return {**state, **marked_page}


def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}


def parse(text: str) -> dict:
    print("Parsed text:\n", text, "\n")
    action_prefix = "Action:"
    last_line = text.strip().split("\n")[-1]
    last_line = re.sub(r"^\s*(\*\*|`)?Action:([^\w]*)", "Action:", last_line)
    if not last_line.startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = last_line

    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}


load_dotenv("/Users/bqhu/langgraph/langgraph-example/.env")
nest_asyncio.apply()

with open("mark_page.js") as f:
    mark_page_script = f.read()

prompt = hub.pull("wfh/web-voyager")
# prompt = prompt_napari
llm = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=4096)
# llm = ModelScopeChatEndpoint(model="Qwen/Qwen2.5-VL-72B-Instruct", max_tokens=4096)
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt_napari | llm | StrOutputParser() | parse
)


def update_scratchpad(state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""
    old = state.get("scratchpad")
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"

    return {**state, "scratchpad": [SystemMessage(content=txt)]}




def select_tool(state: AgentState):
    # Any time the agent completes, this function
    # is called to route the output to a tool or
    # to the end user.
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action



async def main():
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_node("update_scratchpad", update_scratchpad)
    graph_builder.add_edge("update_scratchpad", "agent")

    tools = {
        "Click": click,
        "Type": type_text,
        "Scroll": scroll,
        "Wait": wait,
        "GoBack": go_back,
        "Viewer": to_viewer,
    }


    for node_name, tool in tools.items():
        graph_builder.add_node(
            node_name,
            # The lambda ensures the function's string output is mapped to the "observation"
            # key in the AgentState
            RunnableLambda(tool) | (lambda observation: {"observation": observation}),
        )
        # Always return to the agent (by means of the update-scratchpad node)
        graph_builder.add_edge(node_name, "update_scratchpad")

    graph_builder.add_conditional_edges("agent", select_tool)
    graph = graph_builder.compile()

    async def call_agent(question: str, page, max_steps: int = 150):
        event_stream = graph.astream(
            {
                "page": page,
                "input": question,
                "scratchpad": [],
            },
            {
                "recursion_limit": max_steps,
            },
        )
        final_answer = None
        steps = []

        try:
            async for event in event_stream:
                if "agent" not in event:
                    continue
                pred = event["agent"].get("prediction") or {}
                action = pred.get("action")
                action_input = pred.get("args")

                # 打印交互步骤
                display.clear_output(wait=False)
                steps.append(f"{len(steps)+1}. {action}: {action_input}")
                print("\n".join(steps))
                display.display(display.Image(base64.b64decode(event["agent"]["img"])))

                if "ANSWER" in action:
                    final_answer = action_input[0]
                    break

        finally:
            # 正确关闭异步生成器
            await event_stream.aclose()

        return final_answer



    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    _ = await page.goto("http://127.0.0.1:5020/viewer")
    res = await call_agent("Detect tertiary lymphoid structures(TLS) on slides", page)
    print(f"Final response: {res}")

if __name__ == "__main__":
    asyncio.run(main())
