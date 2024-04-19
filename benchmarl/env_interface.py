class SEDEnv:
    def set_pomg_params(self, designer_action_params):
        """
        Sets the parameters of the POMG based on waht the designer says

        This is named P in the paper

        This function is independent of the reset. Aka you should be able to change the POMG without resetting.

        input -> designer_action_params: the parameters of the POMG given by the action of the designer
        outout -> None

        """
        pass

    def reset(self):
        """
        Normal reset in the instance of the POMG

        input -> None
        outout -> obs, info

        """
        pass

    def step(self, agent_actions):
        """
        Normal step in the instance of the POMG

        input -> agent actions
        outout -> nex obs, next reward, nex info and next done

        """
        pass

    def agent_types(self):
        """

        input -> None
        outout -> The agent types

        """
        pass

    def voting_mechanism(self, agent_types, designer_type):
        """
        Takes as input the agent and designer types and outputs the voting mechanism

        The voting mechiansim is a function that can be called with any quantity that needs to be aggregated
        into a social value, this can be agent rewards or returns for example

        input -> Agent types and designer type
        outout -> the aggregator function representing the social welfare function

        """
        pass

    ###########################
    # Optional but good to have
    ###########################

    def observation(self, agent):
        """

        input -> An agent is the POMG
        outout -> Its observation

        """
        pass

    def state(self):
        """

        input -> None
        outout -> The global state of the POMG

        """
        pass
