FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /

# Install cURL
RUN apt-get update && apt-get install -y curl

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

# Install dependencies
COPY requirements.txt *.py /

RUN pip3 install --no-cache-dir runpod
RUN pip3 install -r /requirements.txt

#Copy the start script
COPY start.sh /

# Make the start script executable
RUN chmod +x /start.sh

# Start the container
CMD ["/start.sh"]