Authentication Service

The system uses JWT tokens for authentication.

When a user logs in:
1. The API checks username and password
2. A JWT token is generated
3. The token is sent back to the client

Middleware verifies the token for protected routes.