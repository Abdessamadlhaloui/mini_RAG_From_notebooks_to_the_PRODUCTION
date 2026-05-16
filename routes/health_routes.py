from fastapi import APIRouter
router = APIRouter(tags=['Health'])
@router.get('/health', summary='Health Check', description='Returns the current health status of the API. Used by load balancers and uptime monitors.', responses={200: {'description': 'Service is healthy'}})
async def health_check():
                                                 
    return {'status': 'ok', 'message': 'API is healthy and running.'}