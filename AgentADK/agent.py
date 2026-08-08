from google.adk.agents import Agent


PRODUCTS = [
    {
        "name": "HP Pavilion 1000",
        "price": 11999,
        "ram": 64,
        "storage": 2048,
        "weight": 2.2,
    },
    {
        "name": "HP Pavilion 1100",
        "price": 17790,
        "ram": 128,
        "storage": 4000,
        "weight": 2.6,
    },
    {
        "name": "Laptop Pro 15",
        "price": 3500,
        "ram": 16,
        "storage": 512,
        "weight": 1.8,
    },
    {
        "name": "Laptop Pro 15 Plus",
        "price": 4500,
        "ram": 16,
        "storage": 1024,
        "weight": 1.8,
    },
    {
        "name": "Laptop Pro 15 Student",
        "price": 3600,
        "ram": 16,
        "storage": 512,
        "weight": 1.7,
    },
    {
        "name": "Laptop XiLiang 120",
        "price": 3850,
        "ram": 16,
        "storage": 1024,
        "weight": 1.8,
    },
    {
        "name": "Laptop Air 14 Work",
        "price": 4200,
        "ram": 16,
        "storage": 512,
        "weight": 1.3,
    },
    {
        "name": "Laptop Air 14a",
        "price": 4000,
        "ram": 16,
        "storage": 800,
        "weight": 1.3,
    },
    {
        "name": "Laptop Basic 15",
        "price": 2600,
        "ram": 8,
        "storage": 256,
        "weight": 2.1,
    },
    {
        "name": "Laptop Basic 15b",
        "price": 2700,
        "ram": 8,
        "storage": 384,
        "weight": 2.1,
    },
    {
        "name": "Laptop Ultra 14",
        "price": 4400,
        "ram": 32,
        "storage": 1100,
        "weight": 1.5,
    },
    {
        "name": "Laptop Ultra 13",
        "price": 3490,
        "ram": 16,
        "storage": 1048,
        "weight": 1.4,
    },
]


def search_products(
    max_price: int | None = None,
    min_ram: int | None = None,
    min_storage: int | None = None,
    max_weight: float | None = None,
):
    """
    Caută produse în funcție de preț, RAM, spațiu de stocare și greutate.

    Args:
        max_price: Prețul maxim acceptat în lei.
        min_ram: Cantitatea minimă de RAM dorită în GB.
        min_storage: Spațiul minim de stocare dorit în GB.
        max_weight: Greutatea maximă acceptată în kg.

    Returns:
        Lista produselor care respectă toate criteriile date.
    """

    results = PRODUCTS

    if max_price is not None:
        results = [p for p in results if p["price"] <= max_price]

    if min_ram is not None:
        results = [p for p in results if p["ram"] >= min_ram]

    if min_storage is not None:
        results = [p for p in results if p["storage"] >= min_storage]

    if max_weight is not None:
        results = [p for p in results if p["weight"] <= max_weight]

    return results


root_agent = Agent(
    name="shopping_agent",
    model="gemini-flash-latest",
    instruction="""
    Ești un Shopping Agent specializat în laptopuri.

    Ajută utilizatorul să găsească produsul potrivit
    pentru nevoile lui.

    Mai întâi află, dacă e relevant:
    - bugetul maxim;
    - memoria RAM minimă dorită;
    - spațiul de stocare minim dorit;
    - greutatea maximă acceptată (dacă portabilitatea contează).

    Nu e nevoie să ceri toate criteriile deodată — poți căuta
    și cu informații parțiale, apoi rafina căutarea dacă e nevoie.

    Când utilizatorul oferă cel puțin un criteriu,
    folosește tool-ul search_products pentru a căuta
    produsele disponibile.

    Nu inventa produse, prețuri sau specificații care nu
    apar în rezultatele tool-ului.

    După ce primești rezultatele de la tool,
    explică utilizatorului care produse se potrivesc
    cel mai bine și de ce (ex. cel mai ieftin, cel mai ușor,
    cel mai performant).

    Dacă nu există niciun produs potrivit, spune asta clar
    și sugerează relaxarea unui criteriu (ex. buget mai mare).
    """,
    tools=[search_products],
)


if __name__ == "__main__":
    import asyncio
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    async def main():
        runner = InMemoryRunner(agent=root_agent)
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="test_user"
        )

        mesaj = "Caut un laptop cu buget maxim 4000 lei și minim 16GB RAM."
        print(f"\nUtilizator: {mesaj}\n")

        content = types.Content(role="user", parts=[types.Part(text=mesaj)])

        async for event in runner.run_async(
            user_id="test_user",
            session_id=session.id,
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent: {part.text}")
                    if part.function_call:
                        print(f"[Apel tool: {part.function_call.name}({dict(part.function_call.args)})]")
                    if part.function_response:
                        print(f"[Rezultat tool: {part.function_response.response}]")

    asyncio.run(main())
