import inspect
import os
from typing import Callable, TypeVar

import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import (add_script_run_ctx,
                                            get_script_run_ctx)


# Define a function to create a callback handler for Streamlit that updates the UI dynamically
def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """
    Creates a Streamlit callback handler that updates the provided Streamlit container with new tokens.
    Args:
        parent_container (DeltaGenerator): The Streamlit container where the text will be rendered.
    Returns:
        BaseCallbackHandler: An instance of a callback handler configured for Streamlit.
    """

    # Define a custom callback handler class for managing and displaying stream events in Streamlit
    class StreamHandler(BaseCallbackHandler):
        """
        Custom callback handler for Streamlit that updates a Streamlit container with new tokens.
        """

        def __init__(
            self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
        ):
            """
            Initializes the StreamHandler with a Streamlit container and optional initial text.
            Args:
                container (st.delta_generator.DeltaGenerator): The Streamlit container where text will be rendered.
                initial_text (str): Optional initial text to start with in the container.
            """
            self.container = container
            self.text = initial_text
            self.run_id_ignore_token = None
            self.instance_id = os.urandom(4).hex()

        def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
            # Workaround to prevent showing the rephrased question as output
            if prompts[0].startswith("Human"):
                self.run_id_ignore_token = kwargs.get("run_id")

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            """
            Callback method triggered when a new token is received (e.g., from a language model).
            Args:
                token (str): The new token received.
                **kwargs: Additional keyword arguments.
            """
            st.session_state.spinner_active = False
            if self.run_id_ignore_token == kwargs.get("run_id"):
                return
            self.text += token
            self.container.markdown(self.text)

    # Define a type variable for generic type hinting in the decorator, to maintain
    # input function and wrapped function return type
    fn_return_type = TypeVar("fn_return_type")

    # Decorator function to add the Streamlit execution context to a function
    def add_streamlit_context(
        fn: Callable[..., fn_return_type]
    ) -> Callable[..., fn_return_type]:
        """
        Decorator to ensure that the decorated function runs within the Streamlit execution context.
        Args:
            fn (Callable[..., fn_return_type]): The function to be decorated.
        Returns:
            Callable[..., fn_return_type]: The decorated function that includes the Streamlit context setup.
        """
        ctx = (
            get_script_run_ctx()
        )  # Retrieve the current Streamlit script execution context

        def wrapper(*args, **kwargs) -> fn_return_type:
            """
            Wrapper function that adds the Streamlit context and then calls the original function.
            Args:
                *args: Positional arguments to pass to the original function.
                **kwargs: Keyword arguments to pass to the original function.
            Returns:
                fn_return_type: The result from the original function.
            """
            add_script_run_ctx(
                ctx=ctx
            )  # Add the Streamlit context to the current execution
            return fn(*args, **kwargs)  # Call the original function with its arguments

        return wrapper

    # Create an instance of the custom StreamHandler with the provided Streamlit container
    st_cb = StreamHandler(parent_container)

    # Iterate over all methods of the StreamHandler instance
    for method_name, method_func in inspect.getmembers(
        st_cb, predicate=inspect.ismethod
    ):
        if method_name.startswith("on_"):  # Identify callback methods
            setattr(
                st_cb, method_name, add_streamlit_context(method_func)
            )  # Wrap and replace the method

    # Return the fully configured StreamHandler instance with the context-aware callback methods
    return st_cb
