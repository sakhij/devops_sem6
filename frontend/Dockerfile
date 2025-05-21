# Use Nginx (a fast web server)
FROM nginx:alpine

# Remove default website
RUN rm -rf /usr/share/nginx/html/*

# Copy your frontend code into the image
COPY . /usr/share/nginx/html

# Expose port 80
EXPOSE 80
