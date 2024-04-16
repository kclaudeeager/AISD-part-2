# AI System Design, HTTP and Web Resources

HTTP (Hypertext Transfer Protocol) is the foundational protocol that enables seamless communication between clients (web browsers, mobile apps, etc.) and servers over the internet. It facilitates the exchange of various resources, such as HTML documents, images, videos, and data, making it the backbone of the modern web.

## Client-Server Architecture

HTTP follows a client-server model, where clients initiate requests to servers, and servers respond with the requested resources or appropriate actions.

- Clients can be web browsers, mobile apps, or any device capable of making HTTP requests.
- Servers host and manage the resources, processing client requests and returning the appropriate responses.

## Request-Response Cycle

The HTTP process operates through a request-response cycle:

1. **Request**: The client sends an HTTP request to the server, specifying the desired resource or action using methods like GET (retrieve data), POST (submit data), PUT (update data), DELETE (remove data), etc.
2. **Response**: The server processes the request and sends back an HTTP response, including status codes, headers, and optionally, the requested resource or data.

## Status Codes

HTTP responses include standardized status codes to indicate the outcome of the request, such as:

- 200 OK: Successful request.
- 201 Created: Request fulfilled, resulting in a new resource creation.
- 400 Bad Request: Client error, such as malformed request syntax.
- 403 Forbidden: Server understands the request but lacks permission to access the resource.
- 404 Not Found: Requested resource not found on the server.
- 500 Internal Server Error: Unexpected server condition preventing request fulfillment.

## URL, URN, and URI

- **URL (Uniform Resource Locator)**: A specific type of URI that provides the address for locating resources on the web, typically consisting of a protocol (e.g., http://), domain name, and path.
- **URN (Uniform Resource Name)**: A persistent and location-independent identifier for resources, designed for long-term reference regardless of changes in resource location or access methods.
- **URI (Uniform Resource Identifier)**: A generic term encompassing both URLs and URNs, serving as a unique identifier for resources.

## APIs and Clients

APIs (Application Programming Interfaces) enable programmatic interaction between clients and server resources, defining the methods and data formats for communication.

Clients can be web browsers, mobile apps, command-line tools, or IoT devices, utilizing APIs to send requests and process responses.

## Resource Assembly

When a client requests a web page, the server responds with an HTML document containing references to additional resources (images, stylesheets, scripts). The client then retrieves these resources through separate HTTP requests, assembling them to render the complete web page for the user.

HTTP is the universal language that enables the seamless exchange of resources and data on the web. Understanding its client-server architecture, request-response cycle, status codes, APIs, and resource assembly is crucial for building and interacting with web-based systems effectively.