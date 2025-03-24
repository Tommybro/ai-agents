from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from pydantic import BaseModel


class Outline(BaseModel):
    """Outline of the book"""
    total_chapters: int
    titles: list[str]


@CrewBase
class OutlineCrew:
    """Outline Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    llm = LLM(
        model="ollama/deepseek-r1:1.5b",
        base_url="http://localhost:11434"
    )

    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["research_agent"],
            llm=self.llm,
            tools=[SerperDevTool()],
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
        )

    @agent
    def outline_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["outline_writer"],
            llm=self.llm,
        )

    @task
    def write_outline(self) -> Task:
        return Task(
            config=self.tasks_config["write_outline"],
            output_pydantic=Outline,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
