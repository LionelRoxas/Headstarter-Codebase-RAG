import ast

class PythonParser:
    def __init__(self):
        # Constructor for the parser class (currently unused)
        pass

    def convert_python_to_ast(self, file_content):
        """
        Converts Python source code into its Abstract Syntax Tree (AST) representation.

        Args:
            file_content (str): The content of the Python file as a string.

        Returns:
            dict: A dictionary containing the AST representation of the provided Python code
              in a string format with indentation for readability.
        """
        # Converts Python source code to AST
        tree = ast.parse(file_content)
        # Turns ast object into a more readable dictionary
        tree_string_format = ast.dump(tree, indent=4)
        return {
            "ast_representation": tree_string_format
        }

    def process_node(self, node, line_start, line_end, largest_size, largest_enclosing_context):
        """
        Check if the given AST node encompasses the specified line range (line_start to line_end).
        Update the largest enclosing context if the current node is a better match.

        Parameters:
        - node: The current AST node being processed.
        - line_start: The start of the target line range.
        - line_end: The end of the target line range.
        - largest_size: The size of the largest context found so far.
        - largest_enclosing_context: The AST node for the largest context found so far.

        Returns:
        - Updated largest_size and largest_enclosing_context.
        """
        # Ensure the node has line number attributes (some nodes might not have these)
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            start = node.lineno  # Start line of the node
            end = node.end_lineno  # End line of the node

            # Check if the node fully encompasses the specified line range
            if start <= line_start and line_end <= end:
                size = end - start  # Calculate the size of the node's range
                if size > largest_size:  # Update if this node's range is larger
                    largest_size = size
                    largest_enclosing_context = node
        return largest_size, largest_enclosing_context

    def find_enclosing_context(self, file_content, line_start, line_end):
        """
        Find the largest enclosing context (AST node) for a specified line range.

        Parameters:
        - file_content: The source code to analyze.
        - line_start: The start of the target line range.
        - line_end: The end of the target line range.

        Returns:
        - A dictionary with the size of the largest context and its AST representation.
        """
        # Parse the source code into an abstract syntax tree (AST)
        tree = ast.parse(file_content)
        largest_size = 0  # Initialize the size of the largest context
        largest_enclosing_context = None  # Initialize the largest enclosing context

        # Traverse the AST using ast.walk (visits all nodes in the tree)
        for node in ast.walk(tree):
            # Process each node to check if it matches the criteria
            largest_size, largest_enclosing_context = self.process_node(
                node, line_start, line_end, largest_size, largest_enclosing_context
            )

        # Return the largest enclosing context and its size
        return {
            "largest_size": largest_size,
            "largest_enclosing_context": ast.dump(largest_enclosing_context) if largest_enclosing_context else None
        }
    

# Example Usage
if __name__ == "__main__":
    # Example Python code to parse
    file_content = """
def foo():
    for i in range(10):
        print(i)
    return 42

def bar():
    print("Hello, World!")
"""

    # Create an instance of the parser
    parser = PythonParser()
    # Find the enclosing context for lines 3 to 4 in the example code
    result = parser.find_enclosing_context(file_content, 3, 4)
    # print(result)  # Print the result of the analysis

    result = parser.convert_python_to_ast(file_content)['ast_representation']
    print("\n\n",result)
