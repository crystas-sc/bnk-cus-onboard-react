import { NextResponse } from 'next/server';

export async function POST(request: Request) {
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
            return NextResponse.json({ error: 'Error from API' }, { status: 500 });
        }
        let data = null;

        try {
            data = await response.json();
            console.log("data", data);
            return NextResponse.json(data);

        }
        catch (error) {
            console.error('Error parsing JSON:', error);
            return NextResponse.json({ error: 'Error parsing JSON' }, { status: 500 });
        }

        

    } catch (error) {
        console.error('API /chat Error:', error);
        return NextResponse.error();
    }
}