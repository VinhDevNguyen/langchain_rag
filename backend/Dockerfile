# Use the official Python image as a base image
FROM python:3.12.3-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libpq-dev

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Django project files into the container
COPY . /app/

# Expose port 3000 to the outside world
EXPOSE 8000

# Run the Django development server

#Run Django server instantly (Production testing)
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

#Sleep infinity for development purposes
CMD ["bash", "-c", "sleep infinity"]