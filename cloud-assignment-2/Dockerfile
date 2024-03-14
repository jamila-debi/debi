# Use Python 3.9 as base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the Python scripts and CSV files into the container
COPY train.py test.py HelloWorld.py train.csv ./

# Install dependencies
RUN pip install pandas scikit-learn joblib

# Run the train.py script
CMD ["python", "train.py", "train.csv"]