# Auto-Wiki: Research Paper Summarization Wiki

Auto-Wiki is a Python project that leverages the power of GPT-4, auto-gpt, and babyAGI (generative LLM-based agents) to automatically summarize and compare research papers into a wiki format. With the rapid growth of research papers being published, this tool aims to make it easier for the community to stay up-to-date and contribute to the knowledge base.

Users can contribute by providing their OpenAI API keys to generate more document summarizations or by submitting code. This collaborative approach ensures that the generated wiki remains relevant and comprehensive.

## Features

- Automatic summarization of research papers
- Comparison of research papers
- Wiki-style format for easy navigation and understanding
- Community-driven contributions
- Powered by GPT-4, auto-gpt, and babyAGI

## Installation

Auto-Wiki uses Poetry for dependency management. To install the project, follow these steps:

1. Install [Poetry](https://python-poetry.org/docs/#installation) if you haven't already.

2. Clone the repository:

   ```
   git clone https://github.com/yourusername/auto-wiki.git
   ```

3. Navigate to the project directory:

   ```
   cd auto-wiki
   ```

4. Install the dependencies using Poetry:

   ```
   poetry install
   ```

## Usage

1. Obtain an OpenAI API key from [OpenAI](https://beta.openai.com/signup/).

2. Export your OpenAI API key as an environment variable:

   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

3. Run the main script with the desired input:

   ```
   poetry run python auto_wiki/run.py https://arxiv.org/pdf/link.to.paper.pdf
   ```

4. The summarized and compared research paper will be generated in the `output` directory.

## Contributing

We welcome contributions from the community! You can contribute by providing your OpenAI API key to generate more documentation for specific documents, or by submitting code or summaries.

1. Fork the repository and create your branch from `main`.

2. Make your changes and commit them.

3. Push your branch and create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- GPT-4, auto-gpt, and babyAGI for providing the generative LLM-based agents
- OpenAI for their API and support (This document was written by gpt-4 afterall)
- The research community for their continuous contributions and collaboration

