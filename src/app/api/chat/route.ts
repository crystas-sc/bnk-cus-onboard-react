import { NextResponse } from 'next/server';

export async function POST(request: Request) {
    try {
        // Send a minimal 202 Accepted response immediately
        const responseInit = { status: 202, statusText: 'Accepted' };
        const responseHeaders = new Headers({ 'Content-Type': 'application/json' });
        const immediateResponse = new Response(JSON.stringify({ message: 'Headers sent' }), { ...responseInit, headers: responseHeaders });

        // Process the request in the background
        (async () => {
            try {
                const body = await request.json();

                const response = await fetch('http://localhost:8000/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(body)
                });

                if (!response.ok) {
                    console.error('API /chat Error:', await response.text());
                    return;
                }

                let data = null;
                try {
                    data = await response.json();
                    console.log("data", data);
                } catch (error) {
                    console.error('Error parsing JSON:', error);
                }
            } catch (error) {
                console.error('API /chat Error:', error);
            }
        })();

        return immediateResponse;
    } catch (error) {
        console.error('API /chat Error:', error);
        return NextResponse.error();
    }
}