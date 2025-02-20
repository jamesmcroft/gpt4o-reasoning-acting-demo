from helpers.base_agent import BaseAgent, skill
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from typing import Optional, List
import json
from pydantic import BaseModel, Field
from helpers.storage_helpers import CustomEncoder, create_json_file
import os
import numpy as np


class Recipe(BaseModel):
    name: str = Field(description="The name of the recipe.")
    author: Optional[str] = Field(
        description="The author of the recipe, if available.")
    ingredients: List[str] = Field(
        description="The ingredients required for the recipe.")
    steps: List[str] = Field(description="The steps to prepare the recipe.")
    embedding: Optional[List[float]] = Field(
        description="The embedding of the recipe for similarity matching. This field must be left as an empty array.")

    def model_dump_markdown(self):
        return f"""
        # Recipe: {self.name}
        ## Ingredients:
        {"".join([f"- {ingredient}\n" for ingredient in self.ingredients])}
        ## Steps:
        {"".join([f"{i+1}. {step}\n" for i, step in enumerate(self.steps)])}
        """


class RecipeAgent(BaseAgent):
    NAME = "Recipe Agent"
    DESCRIPTION = "An agent that can help with cooking recipes."

    def __init__(self, client: OpenAI, model_deployment: str, embedding_model_deployment: str):
        super().__init__(self.NAME, self.DESCRIPTION, client, model_deployment)
        self.embedding_model_deployment = embedding_model_deployment
        self.recipes = [
            Recipe(
                name="Classic Margherita Pizza",
                author="James Croft",
                ingredients=[
                    "1 pizza dough ball",
                    "½ cup pizza sauce",
                    "1 cup shredded mozzarella cheese",
                    "Fresh basil leaves",
                    "Olive oil",
                    "Salt and pepper to taste"
                ],
                steps=[
                    "Preheat your oven to 475°F (245°C) and place a pizza stone inside to heat up.",
                    "Roll out the pizza dough on a floured surface to your desired thickness.",
                    "Spread the pizza sauce over the dough, leaving a small border around the edges.",
                    "Sprinkle the shredded mozzarella cheese over the sauce.",
                    "Bake the pizza on the preheated stone for 10-12 minutes or until the crust is golden and the cheese is bubbly.",
                    "Remove the pizza from the oven and top with fresh basil leaves, a drizzle of olive oil, and salt and pepper to taste."
                ],
                embedding=None
            ),
            Recipe(
                name="Eggs Benedict",
                author="James Croft",
                ingredients=[
                    "4 eggs",
                    "2 English muffins, split",
                    "4 slices Canadian bacon",
                    "Hollandaise sauce",
                    "Salt and pepper to taste",
                    "Chopped parsley (for garnish)"
                ],
                steps=[
                    "Fill a large saucepan with 2-3 inches of water and bring to a simmer.",
                    "In a separate saucepan, heat the Hollandaise sauce over low heat, stirring occasionally.",
                    "Toast the English muffins and cook the Canadian bacon in a skillet until heated through.",
                    "Poach the eggs in the simmering water for 3-4 minutes until the whites are set but the yolks are still runny.",
                    "Assemble the Eggs Benedict by placing a slice of Canadian bacon on each English muffin half, topping with a poached egg, and drizzling with Hollandaise sauce.",
                    "Season with salt and pepper, and garnish with chopped parsley before serving."
                ],
                embedding=None
            ),
            Recipe(
                name="Vegan Chocolate Cake",
                author="James Croft",
                ingredients=[
                    "1 ½ cups all-purpose flour",
                    "1 cup organic cane sugar",
                    "½ cup cocoa powder",
                    "1 tsp baking soda",
                    "½ tsp salt",
                    "1 cup almond milk",
                    "⅓ cup vegetable oil",
                    "1 tbsp apple cider vinegar",
                    "1 tsp vanilla extract"
                ],
                steps=[
                    "Preheat your oven to 350°F (175°C) and grease an 8-inch round cake pan.",
                    "In a large bowl, sift together flour, sugar, cocoa powder, baking soda, and salt.",
                    "In a separate bowl, whisk almond milk, vegetable oil, apple cider vinegar, and vanilla extract.",
                    "Pour the wet ingredients into the dry ingredients and mix until just combined.",
                    "Pour the batter into the prepared pan and bake for 30-35 minutes or until a toothpick inserted in the center comes out clean.",
                    "Let the cake cool in the pan for 10 minutes, then transfer to a wire rack to cool completely."
                ],
                embedding=None
            ),
            Recipe(
                name="Spaghetti Bolognese",
                author="James Croft",
                ingredients=[
                    "400g spaghetti",
                    "500g ground beef",
                    "1 large onion, finely chopped",
                    "3 cloves garlic, minced",
                    "2 cans (400g each) diced tomatoes",
                    "2 tbsp tomato paste",
                    "100ml red wine (optional)",
                    "1 tsp dried oregano",
                    "1 tsp dried basil",
                    "Salt and pepper to taste",
                    "2 tbsp olive oil",
                    "Grated Parmesan cheese (for serving)"
                ],
                steps=[
                    "Cook the spaghetti according to package instructions until al dente. Drain and set aside.",
                    "Heat olive oil in a large pan over medium heat. Add chopped onion and garlic; sauté until soft and translucent.",
                    "Add ground beef and cook until browned, breaking it up with a spoon as it cooks.",
                    "Stir in tomato paste and cook for 1-2 minutes to develop the flavor.",
                    "Pour in diced tomatoes and red wine, then add oregano and basil. Season with salt and pepper.",
                    "Bring the sauce to a simmer and let it cook for 20-30 minutes, stirring occasionally.",
                    "Serve the sauce over spaghetti and top with grated Parmesan cheese."
                ],
                embedding=None
            ),
            Recipe(
                name="Beef Stir-Fry with Vegetables",
                author="James Croft",
                ingredients=[
                    "400g beef sirloin, thinly sliced",
                    "1 head broccoli, cut into florets",
                    "1 red bell pepper, sliced",
                    "1 yellow bell pepper, sliced",
                    "2 carrots, julienned",
                    "100g snap peas",
                    "2 cloves garlic, minced",
                    "1 tbsp fresh ginger, grated",
                    "3 tbsp soy sauce",
                    "1 tbsp sesame oil",
                    "1 tsp cornstarch mixed with 2 tsp water",
                    "2 tbsp vegetable oil",
                    "Cooked rice (for serving)"
                ],
                steps=[
                    "Marinate the beef slices in 2 tbsp soy sauce, garlic, and ginger for 15 minutes.",
                    "Heat vegetable oil in a wok or large pan over high heat. Stir-fry the beef until nearly cooked through, then remove from the pan.",
                    "In the same pan, add a little more oil if needed, and stir-fry the broccoli, bell peppers, carrots, and snap peas for 3-4 minutes until crisp-tender.",
                    "Return the beef to the pan, add the remaining 1 tbsp soy sauce and sesame oil, and stir-fry for another 2 minutes.",
                    "Pour in the cornstarch slurry to thicken the sauce slightly, stirring well.",
                    "Serve hot over a bed of cooked rice."
                ],
                embedding=None
            ),
            Recipe(
                name="Strawberry Cheesecake",
                author="James Croft",
                ingredients=[
                    "For the crust:",
                    "1 ½ cups graham cracker crumbs (vegan)",
                    "5 tbsp melted coconut oil",
                    "2 tbsp maple syrup",
                    "For the filling:",
                    "3 cups raw cashews (soaked overnight, drained)",
                    "1 cup coconut cream",
                    "¾ cup maple syrup",
                    "¼ cup lemon juice",
                    "1 tsp vanilla extract",
                    "For the topping:",
                    "2 cups fresh strawberries, sliced",
                    "2 tbsp strawberry jam"
                ],
                steps=[
                    "Preheat your oven to 350°F (175°C).",
                    "For the crust, mix graham cracker crumbs, melted coconut oil, and maple syrup. Press firmly into the bottom of a springform pan.",
                    "Bake the crust for 8-10 minutes, then let cool.",
                    "For the filling, blend soaked cashews, coconut cream, maple syrup, lemon juice, and vanilla extract until completely smooth.",
                    "Pour the filling over the cooled crust and spread evenly.",
                    "Bake in a water bath for 25-30 minutes, then cool to room temperature before refrigerating for at least 4 hours.",
                    "For the topping, mix sliced strawberries with strawberry jam and arrange them on top of the cheesecake before serving."
                ],
                embedding=None
            ),
            Recipe(
                name="Vegan Banana Bread",
                author="James Croft",
                ingredients=[
                    "3 overripe bananas, mashed",
                    "1/3 cup melted coconut oil",
                    "1 cup brown sugar",
                    "1 tsp vanilla extract",
                    "1 tsp baking soda",
                    "Pinch of salt",
                    "1 ½ cups all-purpose flour",
                    "1/2 cup chopped walnuts (optional)"
                ],
                steps=[
                    "Preheat your oven to 350°F (175°C) and grease a 9x5-inch loaf pan.",
                    "In a large bowl, mix mashed bananas with melted coconut oil, brown sugar, and vanilla extract.",
                    "Sprinkle in the baking soda and salt, stirring to combine.",
                    "Gently fold in the flour and walnuts until just incorporated.",
                    "Pour the batter into the loaf pan and smooth the top.",
                    "Bake for 50-60 minutes or until a toothpick inserted in the center comes out clean.",
                    "Allow the bread to cool in the pan for 10 minutes, then transfer to a wire rack."
                ],
                embedding=None
            ),
            Recipe(
                name="Chicken Tikka Masala",
                author="James Croft",
                ingredients=[
                    "For the chicken marinade:",
                    "1 lb boneless, skinless chicken thighs, cut into bite-sized pieces",
                    "1 cup plain yogurt",
                    "2 tbsp lemon juice",
                    "2 tsp ground cumin",
                    "2 tsp paprika",
                    "1 tsp ground cinnamon",
                    "1 tsp ground cayenne pepper",
                    "1 tsp ground black pepper",
                    "1 tsp salt",
                    "For the sauce:",
                    "2 tbsp vegetable oil",
                    "1 large onion, finely chopped",
                    "3 cloves garlic, minced",
                    "1 tbsp fresh ginger, grated",
                    "1 tbsp garam masala",
                    "1 tsp ground turmeric",
                    "1 tsp ground coriander",
                    "1 tsp ground cumin",
                    "1 can (400g) crushed tomatoes",
                    "1 cup coconut milk",
                    "Salt and pepper to taste",
                    "Fresh cilantro (for garnish)"
                ],
                steps=[
                    "In a large bowl, combine chicken pieces with yogurt, lemon juice, and spices for the marinade. Cover and refrigerate for at least 1 hour.",
                    "Heat vegetable oil in a large pan over medium heat. Add chopped onion, garlic, and ginger; sauté until soft and fragrant.",
                    "Add garam masala, turmeric, coriander, and cumin to the pan; cook for 1-2 minutes to toast the spices.",
                    "Stir in crushed tomatoes and coconut milk, then season with salt and pepper.",
                    "Add marinated chicken to the sauce and simmer for 20-30 minutes until the chicken is cooked through.",
                    "Serve the Chicken Tikka Masala over rice, garnished with fresh cilantro."
                ],
                embedding=None
            ),
            Recipe(
                name="Vegetable Fried Rice",
                author="James Croft",
                ingredients=[
                    "2 cups cooked rice, chilled",
                    "1 cup mixed vegetables (peas, carrots, corn, etc.)",
                    "2 eggs, beaten",
                    "2 cloves garlic, minced",
                    "2 tbsp soy sauce",
                    "1 tbsp sesame oil",
                    "1 tbsp vegetable oil",
                    "Salt and pepper to taste",
                    "Green onions (for garnish)"
                ],
                steps=[
                    "Heat vegetable oil in a large pan or wok over medium heat. Add minced garlic and cook until fragrant.",
                    "Push the garlic to the side of the pan and pour in the beaten eggs. Scramble the eggs until cooked through.",
                    "Add mixed vegetables to the pan and stir-fry until heated through.",
                    "Stir in the chilled rice, breaking up any clumps with a spatula.",
                    "Drizzle soy sauce and sesame oil over the rice, then season with salt and pepper.",
                    "Continue to stir-fry the rice until everything is well combined and heated through.",
                    "Garnish with chopped green onions before serving."
                ],
                embedding=None
            ),
            Recipe(
                name="Classic Chocolate Chip Cookies",
                author="James Croft",
                ingredients=[
                    "1 cup unsalted butter, softened",
                    "1 cup brown sugar",
                    "½ cup granulated sugar",
                    "2 large eggs",
                    "1 tsp vanilla extract",
                    "2 ½ cups all-purpose flour",
                    "1 tsp baking soda",
                    "½ tsp salt",
                    "2 cups chocolate chips"
                ],
                steps=[
                    "Preheat your oven to 375°F (190°C) and line a baking sheet with parchment paper.",
                    "In a large bowl, cream together butter, brown sugar, and granulated sugar until light and fluffy.",
                    "Beat in eggs one at a time, then stir in vanilla extract.",
                    "In a separate bowl, whisk together flour, baking soda, and salt.",
                    "Gradually add the dry ingredients to the wet ingredients, mixing until just combined.",
                    "Fold in the chocolate chips.",
                    "Drop spoonfuls of dough onto the prepared baking sheet and bake for 8-10 minutes or until golden brown.",
                    "Let the cookies cool on the baking sheet for a few minutes before transferring to a wire rack to cool completely."
                ],
                embedding=None
            ),
            Recipe(
                name="Scrambled Eggs with Spinach and Feta on Toast",
                author="James Croft",
                ingredients=[
                    "4 large eggs",
                    "1 cup baby spinach",
                    "½ cup crumbled feta cheese",
                    "4 slices seeded bread",
                    "2 tbsp butter",
                    "Salt and pepper to taste"
                ],
                steps=[
                    "In a bowl, whisk together eggs, baby spinach, and crumbled feta cheese.",
                    "Heat butter in a non-stick pan over medium heat. Pour in the egg mixture and cook, stirring occasionally, until the eggs are scrambled and cooked through.",
                    "Toast the bread slices until golden brown and crispy.",
                    "Divide the scrambled eggs between the toast slices and season with salt and pepper before serving."
                ],
                embedding=None
            )
        ]

        self._load_recipes()

    def _load_recipes(self):
        save_recipes = False

        if os.path.exists("./recipes.json") and os.path.getsize("./recipes.json") > 0:
            with open("./recipes.json", "r") as f:
                self.recipes = []
                for recipe in json.load(f):
                    self.recipes.append(Recipe(**recipe))

        for recipe in self.recipes:
            if not recipe.embedding:
                recipe.embedding = self._create_recipe_embedding(recipe)
                save_recipes = True

        if save_recipes:
            self._save_recipes()

    def _create_recipe_embedding(self, recipe: Recipe):
        return self._create_embedding(recipe.model_dump_markdown())

    def _save_recipes(self):
        create_json_file("./recipes.json", self.recipes)

    def _create_embedding(self, text: str) -> List[float]:
        embedding_response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model_deployment)
        return embedding_response.data[0].embedding

    @skill
    def find_recipes_by_description(self, description: str, available_ingredients: Optional[List[str]], count: Optional[int] = 1) -> str:
        """
        Find a single recipe that best matches the given description.

        Args:
        - description: A description of the recipe the user is looking for.
        - available_ingredients: An optional list of ingredients that the user has available.
        - count: The number of recipes to return. Default is 1.
        """

        query = f"""Find a recipe that best matches the following description:
        {description}

        Available ingredients: {", ".join(available_ingredients)}
        """

        query_embedding = self._create_embedding(query)

        embeddings_matrix = np.array(
            [recipe.embedding for recipe in self.recipes])

        similarity_scores = np.dot(embeddings_matrix, query_embedding)
        best_match_indices = np.argsort(similarity_scores)[::-1][:count]

        filtered_indices = best_match_indices[similarity_scores[best_match_indices] > 0.5]

        if filtered_indices.size > 0:
            best_matches = [self.recipes[i] for i in filtered_indices]
            return "\n\n".join([recipe.model_dump_markdown() for recipe in best_matches])

        return "Sorry, I couldn't find recipes that matches your description."

    @skill
    def find_ingredients_in_kitchen(self) -> str:
        """
        Find the ingredients that are available in the kitchen.
        """

        # Emulates finding available ingredients in the kitchen.
        return json.dumps([
            "500g plain flour",
            "200g caster sugar",
            "500g cocoa powder",
            "100g pasta",
            "2 cans of tomatoes",
            "A broccoli head",
            "1L maple syrup",
            "200g light brown sugar",
            "100g bicarbonate of soda",
            "1kg walnuts",
            "150ml vanilla extract",
            "500g coconut oil",
            "1kg oats",
            "1kg raisins",
            "500g dark chocolate",
            "150g almonds",
        ], cls=CustomEncoder)

    @skill
    def modify_recipe_if_not_vegan(self, recipe_name: str) -> str:
        """
        Modifies a known recipe to make it vegan-friendly, if it contains meat or dairy products.

        Args:
        - recipe_name: The name of the recipe to modify.
        """

        recipe = next((recipe for recipe in self.recipes if recipe.name.lower(
        ) == recipe_name.lower()), None)

        if recipe:
            messages = [ChatCompletionSystemMessageParam(role="system", content=f"""You are an AI agent that helps with modifying an existing recipe to make it vegan-friendly.
                                                         ## On your ability to modify recipes

                                                         - Replace any meat or dairy products in the recipe with appropriate vegan alternatives.
                                                         - Ensure that the modified recipe retains the essence and flavor of the original recipe.
                                                         - If the recipe is already vegan, you should not make any changes.
                                                         - If the recipe can't be modified to be vegan-friendly, you should return an empty recipe.
                                                         - Use the author name, "Recipe Agent", when providing the modified recipe to denote the modification.
                                                         """),
                        ChatCompletionUserMessageParam(
                            role="user", content=recipe.model_dump_markdown())
                        ]

            completion = self.client.beta.chat.completions.parse(
                model=self.model_deployment,
                messages=messages,
                response_format=Recipe,
                temperature=0.3,
                top_p=0.3,
            )

            if completion.choices[0].message.parsed:
                vegan_recipe = completion.choices[0].message.parsed
                vegan_recipe.embedding = self._create_recipe_embedding(
                    vegan_recipe)
                self.recipes.append(vegan_recipe)
                self._save_recipes()

                return vegan_recipe.model_dump_markdown()
            else:
                return f"Sorry, I couldn't modify the recipe {recipe_name} to be vegan-friendly."

        return f"Sorry, I couldn't find a recipe with the name {recipe_name}."

    @skill
    def generate_shopping_list_from_recipe(self, recipe_name: str, available_ingredients: Optional[List[str]]) -> str:
        """
        Generate a shopping list based on the ingredients required for a recipe and the available ingredients in the kitchen.

        Args:
        - recipe_name: The name of the recipe to generate a shopping list for.
        - available_ingredients: An optional list of ingredients that are available in the kitchen.
        """

        recipe = next((recipe for recipe in self.recipes if recipe.name.lower(
        ) == recipe_name.lower()), None)

        if recipe:
            messages = [ChatCompletionSystemMessageParam(role="system", content=f"""You are an AI agent that helps generate a shopping list based on the ingredients required for a recipe and the available ingredients in the kitchen.
                                                         ## On your ability to generate a shopping list

                                                         - Compare the ingredients required for the recipe with the available ingredients in the kitchen.
                                                         - If there are potential substitutions or alternatives for any ingredients, you should suggest them.
                                                         - Identify the ingredients that are missing and need to be purchased.
                                                         - Provide a list of the missing ingredients that need to be purchased.
                                                         - If all the ingredients are available, you should return an empty shopping list.
                                                         """),
                        ChatCompletionUserMessageParam(
                            role="user", content=[
                                ChatCompletionContentPartTextParam(
                                    type="text", content=recipe.model_dump_markdown()),
                                ChatCompletionContentPartTextParam(
                                    type="text", content=f"""Available ingredients:\n\n{json.dumps(available_ingredients, cls=CustomEncoder)}""")
                            ])
                        ]

            completion = self.client.chat.completions.create(
                model=self.model_deployment,
                messages=messages,
                temperature=0.3,
                top_p=0.3,
            )

            if completion.choices[0].message.content:
                return completion.choices[0].message.content
            else:
                return f"Sorry, I couldn't generate a shopping list for the recipe {recipe_name}."

        return f"Sorry, I couldn't find a recipe with the name {recipe_name}."

    def process_query(self, messages: List[str]) -> ChatCompletionMessage:
        execute_messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=f"""You are an agent that can help with cooking recipes.

                ## On your ability to provide recipes

                - You have access to a bank of recipes that you can provide to users based on their queries.
                - If a recipe is found that contains meat or dairy products, you are allowed to modify the ingredients and recipe of the dish to make it vegan-friendly.
                - If a recipe can't be found based on the user's query, you should not provide a recipe and instead inform the user that you couldn't find a recipe that matches their description.
                - When providing a recipe, **always** return the necessary recipe details, including the name, ingredients, and steps to prepare the dish.
                - If a recipe is found, but is not 100% accurate, still return the recipe details and inform the user that the recipe may need some adjustments.
                
                ## Available skills
                
                You have the following skills available to assist in your tasks:
                
                {"\n".join(
                    [f"- {skill['function']['name']}: {skill['function']['description']}" for skill in self.skills])}
                """
            )
        ]

        execute_messages.extend([message for message in messages])

        function_responses = []

        completion = self.client.chat.completions.create(
            model=self.model_deployment,
            messages=execute_messages,
            temperature=0.3,
            top_p=0.3,
            tools=self.skills,
            tool_choice="auto"
        )

        message = completion.choices[0].message
        execute_messages.append(message)

        if message.tool_calls:
            print(f"Executing {len(message.tool_calls)} tool functions...")

            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"Executing tool function: {
                      function_name} with arguments: {function_args}")

                response = self.call_function(function_name, **function_args)

                execute_messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": response
                })

                function_responses.append(response)

        completion = self.client.chat.completions.create(
            model=self.model_deployment,
            messages=execute_messages,
            temperature=0.3,
            top_p=0.3,
        )

        return ChatCompletionMessage(role="assistant", content=completion.choices[0].message.content)
