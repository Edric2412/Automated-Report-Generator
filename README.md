# Automated Report Generator

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Edric2412/Automated-Report-Generator)

## 📌 Overview

The **Automated Report Generator** is a streamlined web application designed to automate the process of creating professional event reports. It simplifies report creation by integrating structured data from a user-friendly form, reducing manual effort, and enhancing productivity through intelligent features.

This tool is perfect for educational institutions, organizations, and teams that need to generate consistent, high-quality reports for workshops, seminars, and other events.

## ✨ Features

-   **Automated Report Creation:** Generate structured DOCX reports from a single form submission.
-   **User-Friendly Interface:** A modern, intuitive interface for easy data entry and file uploads.
-   **Dynamic Templates:** Automatically adapts the report structure based on the event type.
-   **Document Preview:** Instantly view a live preview of the generated report before downloading.
-   **AI-Powered Summaries:** Automatically generate concise summaries for event descriptions and outcomes using an extractive summarization model.
-   **In-Browser Photo Capture:** Capture action photos directly from your device's camera.
    -   **Geo-tagging & Timestamping:** Automatically overlays captured photos with precise GPS coordinates, address, and a timestamp.
    -   **Mobile-Friendly Controls:** Supports camera switching (front/back) and both portrait and landscape orientations on smartphones.
-   **Multi-Format Support:** Exports final reports in DOCX format for easy editing and sharing.
-   **Secure & Efficient:** Built with FastAPI for quick, reliable processing and data handling.

## 🛠 Installation

Follow these steps to set up the project locally:

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Edric2412/Automated-Report-Generator.git
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd Automated-Report-Generator
    ```

3.  **Create and activate a virtual environment (recommended):**
    ```sh
    # Create the environment
    python -m venv venv

    # Activate it (on Windows)
    venv\Scripts\activate

    # Activate it (on macOS/Linux)
    source venv/bin/activate
    ```

4.  **Install dependencies from `requirements.txt`:**
    ```sh
    pip install -r requirements.txt
    ```

5.  **Download NLTK Data (First-time setup):**
    The application will automatically download the necessary NLP data (`punkt_tab`) on its first run. No manual steps are needed.

## 🚀 Usage

1.  **Run the application server:**
    ```sh
    uvicorn backend.main:app --reload
    ```

2.  **Open the web interface** in your browser by navigating to:
    ```
    http://127.0.0.1:8000
    ```

3.  **Fill out the event details** in the form.
4.  **Use the "Generate AI Summary"** buttons for assistance with the summary and outcome sections.
5.  **Upload event photos** by browsing your files or using the "Take Photo" feature.
6.  Click **"Preview Report"** to see a live version of the document.
7.  Click **"Generate Report"** to create the final DOCX file and receive a download link.

## 🤝 Contributing

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/your-feature-name`).
3.  Commit your changes with clear, descriptive messages (`git commit -m "feat: Added new feature"`).
4.  Push your changes to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request and describe the changes you've made.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any queries, bug reports, or suggestions, please open an issue in the [GitHub Issues](https://github.com/Edric2412/Automated-Report-Generator/issues) section.
