# HTTP API documentation

The Reachy Mini daemon provides a REST API that you can use to control the robot and get its state and even control the daemon itself. The API is implemented using [FastAPI](https://fastapi.tiangolo.com/) and [pydantic](https://docs.pydantic.dev/latest/) models.

It should provide you all the necessary endpoints to interact with the robot, including:
- Getting the state of the robot (joints positions, motor status, etc.)
- Moving the robot's joints or setting specific poses

The API is documented using OpenAPI, and you can access all available routes and test them at [http://localhost:8000/docs](http://localhost:8000/docs) when the daemon is running.
You can also access the raw OpenAPI schema at [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json).

This can be useful if you want to generate client code for your preferred programming language or framework, connect it to your AI application, or even to create your MCP server.

### WebSocket support

The API also supports WebSocket connections for real-time updates. For instance, you can subscribe to joint state updates:

```js
let ws = new WebSocket(`ws://127.0.0.1:8000/api/state/ws/full`);

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
};
```