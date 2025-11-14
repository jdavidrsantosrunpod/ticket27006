# ac-sl-py-mcp-autopsy

## Connecting to the Deployed Service

This service has been deployed as a serverless pod on RunPod with queue type configuration.

### Pod Information
- **Endpoint ID**: `blablabla`
- **Deployment Type**: Serverless Queue
- **Platform**: RunPod

### How to Connect

#### Request a new Job (/run)
Asynchronous jobs process in the background and return immediately with a job ID. This approach works best for longer-running tasks that don’t require immediate results, operations requiring significant processing time, and managing multiple concurrent jobs. /run requests have a maximum payload size of 10 MB. Job results are available for 30 minutes after completion.


    curl --request POST \
        --url https://api.runpod.ai/v2/$ENDPOINT_ID/run \
        -H "accept: application/json" \
        -H "authorization: $RUNPOD_API_KEY" \
        -H "content-type: application/json" \
        -d '{"input": {"prompt": "Hello, world!"}}'

`/run` returns a response with the job ID and status:

    {
    "id": "eaebd6e7-6a92-4bb8-a911-f996ac5ea99d",
    "status": "IN_QUEUE"
    }

Further results must be retrieved using the /status operation.
​
#### Check status and fetch results (/status)
Check the current state, execution statistics, and results of previously submitted jobs. The status operation provides the current job state, execution statistics like queue delay and processing time, and job output if completed.
You can configure time-to-live (TTL) for individual jobs by appending a TTL parameter to the request URL.For example, https://api.runpod.ai/v2/$ENDPOINT_ID/status/YOUR_JOB_ID?ttl=6000 sets the TTL to 6 seconds.

Replace `YOUR_JOB_ID` with the actual job ID you received in the response to the `/run` operation.

    curl --request GET \
        --url https://api.runpod.ai/v2/$ENDPOINT_ID/status/YOUR_JOB_ID \
        -H "authorization: $RUNPOD_API_KEY" \

`/status` returns a JSON response with the job status (e.g. `IN_QUEUE`, `IN_PROGRESS`, `COMPLETED`, `FAILED`), and an optional `output` field if the job is completed:

    {
    "delayTime": 31618,
    "executionTime": 1437,
    "id": "60902e6c-08a1-426e-9cb9-9eaec90f5e2b-u1",
    "output": {
        "input_tokens": 22,
        "output_tokens": 16,
        "text": ["Hello! How can I assist you today?\nUSER: I'm having"]
    },
    "status": "COMPLETED"
    }
