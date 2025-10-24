# Chatbot Implementation Module

This module provides two distinct implementations of a conversational agent, colloquially known as a "chatbot," leveraging our fine-tuned Large Language Model (LLM), `Formal-LLaMAntino-3`. The primary objective of this module is to serve as a practical demonstration of the model's conversational capabilities, contextual understanding, and adherence to predefined behavioral instructions. The implementations are offered in two modalities: a Command-Line Interface (CLI) and a Graphical User Interface (GUI), to accommodate diverse user preferences and technical environments.

## Module Structure

The module is composed of the following components:

*   `cli_chat_formal_llamantino_3.py`: This script provides a terminal-based chatbot interface. It is engineered for users who prefer a lightweight, text-only interaction. The script manages conversational context through a history mechanism and employs 4-bit quantization for efficient deployment on resource-constrained systems.

*   `gui_chat_formal_llamantino_3.py`: This script instantiates a chatbot with a graphical user interface built upon the Tkinter library. This version offers a more interactive and user-friendly experience. A key feature of the GUI implementation is the real-time configurability of the model's generative parameters, including `temperature`, `top-p`, `top-k`, and `max_new_tokens`. This allows for dynamic experimentation with the model's response generation process.

## Core Functionality

Both implementations share a core set of functionalities:

*   **Model Integration**: They utilize the `Formal-LLaMAntino-3` model, loaded with 4-bit quantization to ensure computational efficiency.
*   **System Prompt**: A carefully crafted system prompt is employed to instruct the model on its persona, objectives, and operational constraints. This foundational message guides the model's behavior, ensuring responses are aligned with the desired formal and helpful assistant persona.
*   **Conversation History**: The context of the conversation is maintained through a list of messages, which is passed to the model with each new user query. This enables the model to generate contextually relevant and coherent responses.

## Usage

To run either of the chatbots, execute the corresponding Python script from the terminal:

For the CLI version:
```bash
python cli_chat_formal_llamantino_3.py
```

For the GUI version:
```bash
python gui_chat_formal_llamantino_3.py
```
*Prerequisites: Ensure all dependencies listed in the main `requirements.txt` are installed.*

## Recommendation

For a comprehensive and interactive evaluation of the model's capabilities, the GUI implementation is recommended. Its interface for real-time parameter adjustment provides a valuable tool for exploring the nuances of the model's text generation process.
