# Use the official Node.js image as the base image
FROM node:20.13.1-bookworm AS build

# Set the working directory
WORKDIR /app

# Copy the package.json and package-lock.json (or yarn.lock) files to the working directory
COPY package*.json ./

# Install the dependencies
RUN npm install

# Copy the rest of the application code to the working directory
COPY . .

#Start the development environment
CMD ["npm", "start"]

### Following lines to build and set up a running application with nginx
# # Build the React application for production
# RUN npm run build

# # Use the official nginx image to serve the built files
# FROM nginx:1.26.0-bookworm

# # Copy the built files from the previous stage to the nginx html directory
# COPY --from=build /app/build /usr/share/nginx/html

# # Expose port 80 to the outside world
# EXPOSE 80

# # Start nginx
# CMD ["nginx", "-g", "daemon off;"]
