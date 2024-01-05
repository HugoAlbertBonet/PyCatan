import random

from Classes.Constants import BuildConstants, HarborConstants
from Classes.DevelopmentCards import DevelopmentCard
from Classes.Hand import Hand
from Classes.TradeOffer import TradeOffer
from Managers.GameManager import GameManager
from TraceLoader.TraceLoader import TraceLoader


class GameDirector:
    """
    Clase que se encarga de dirigir la partida, empezarla y acabarla
    """

    def __init__(self):
        self.game_manager = GameManager()
        self.trace_loader = TraceLoader()
        # self.turn_manager = TurnManager()
        # self.bot_manager = BotManager()
        self.MAX_COMMERCE_TRADES = 2
        self.already_played_development_card = False
        return

    def reset_game_values(self):
        # Reseteamos la traza actual
        self.trace_loader.current_trace = {}

        # Reseteamos el game_manager
        self.game_manager.reset_game_values()
        return

    # -- -- -- --  Turn  -- -- -- --
    def start_turn(self, player=-1):
        """
        Esta función permite iniciar el turno a un jugador.
        :param player: (int) número que representa al jugador.
        :return: object
        """
        start_turn_object = {'development_card_played': []}

        self.game_manager.set_phase(0)
        self.game_manager.set_actual_player(player)

        turn_start_response = self.game_manager.on_turn_start(player)

        if isinstance(turn_start_response, DevelopmentCard) and not self.already_played_development_card:
            played_card_obj = self.game_manager.play_development_card(player, turn_start_response)
            if not (played_card_obj['played_card'] == 'victory_point' or
                    played_card_obj['played_card'] == 'failed_victory_point'):
                self.already_played_development_card = True
            start_turn_object['development_card_played'].append(played_card_obj)

        self.game_manager.throw_dice()
        self.game_manager.give_resources()

        start_turn_object['dice'] = self.game_manager.get_last_dice_roll()
        start_turn_object['actual_player'] = str(self.game_manager.get_whose_turn_is_it())

        # TODO: mover lógica al GameManager dentro del método "check_if_thief_is_called()".
        #  Debe devolver un objeto que hace lo que "start_turn_object" hace aquí para poder agregarlo a start_turn_object aquí
        if self.game_manager.get_last_dice_roll() == 7:
            for obj in self.game_manager.get_players():
                if obj['resources'].get_total() > 7:
                    total = obj['player'].on_having_more_than_7_materials_when_thief_is_called().get_total()
                    max_hand = (total / 2).__floor__()

                    while total > max_hand:
                        obj['resources'].remove_material(random.randint(0, 4), 1)
                        total = obj['resources'].get_total()

            on_moving_thief = self.game_manager.get_players()[player]['player'].on_moving_thief()
            move_thief_obj = self.game_manager.move_thief(on_moving_thief['terrain'], on_moving_thief['player'])

            start_turn_object['past_thief_terrain'] = move_thief_obj['last_thief_terrain']
            start_turn_object['thief_terrain'] = move_thief_obj['terrain_id']
            start_turn_object['robbed_player'] = move_thief_obj['robbed_player']
            start_turn_object['stolen_material_id'] = move_thief_obj['stolen_material_id']
        # TODO: fin de mover lógica a "check_if_thief_is_called"

        for i in range(4):
            start_turn_object['hand_P' + str(i)] = self.game_manager.player_resources_to_object(i)
            start_turn_object['total_P' + str(i)] = str(self.game_manager.player_resources_total(i))

        return start_turn_object

    def start_commerce_phase(self, player=-1, depth=1):
        """
        Esta función permite pasar a la fase de comercio a un jugador.
        :param depth: (int) número de veces que ha comerciado ya el jugador.
        :param player: (int) número que representa al jugador.
        :return: object
        """
        commerce_phase_object = {}

        self.game_manager.set_phase(1)

        commerce_response = self.game_manager.call_to_bot_on_commerce_phase(player)

        # TODO: mover lógica a self.game_manager.on_commerce_response() -> devuelve un objeto similar a "commerce_phase_object"
        if isinstance(commerce_response, TradeOffer) and depth <= self.MAX_COMMERCE_TRADES:
            commerce_phase_object['trade_offer'] = commerce_response.__to_object__()
            commerce_phase_object['harbor_trade'] = False

            # TODO: commerce_response siempre será True dado que está llegando ya como TradeOffer
            if commerce_response:

                if self.game_manager.get_players()[player]['resources'].resources.has_this_more_materials(
                        commerce_response.gives):
                    commerce_phase_object['inviable'] = False
                    answer_object = self.game_manager.send_trade_to_everyone(commerce_response)
                    commerce_phase_object['answers'] = answer_object
                else:
                    commerce_phase_object['inviable'] = True
                    # TODO: se queja de que no puede hacerla, le da una segunda oportunidad, en otro fallo
                    #       le salta la fase de comercio

            return commerce_phase_object
        elif isinstance(commerce_response, dict):

            commerce_phase_object['trade_offer'] = commerce_response
            commerce_phase_object['harbor_trade'] = True

            harbor_type = self.game_manager.board.check_for_player_harbors(player, commerce_response['gives'])
            if harbor_type == HarborConstants.NONE:
                response = self.game_manager.commerce_manager.trade_without_harbor(
                    self.game_manager.get_players()[player]['resources'], commerce_response['gives'],
                    commerce_response['receives'])
            elif harbor_type == HarborConstants.ALL:
                response = self.game_manager.commerce_manager.trade_through_harbor(
                    self.game_manager.get_players()[player]['resources'], commerce_response['gives'],
                    commerce_response['receives'])
            else:
                response = self.game_manager.commerce_manager.trade_through_special_harbor(
                    self.game_manager.get_players()[player]['resources'], commerce_response['gives'],
                    commerce_response['receives'])

            if isinstance(response, Hand):
                commerce_phase_object['answer'] = response.resources.__to_object__()
                self.game_manager.get_players()[player]['player'].hand = response
            else:
                commerce_phase_object['answer'] = response
            return commerce_phase_object
        elif isinstance(commerce_response, DevelopmentCard) and not self.already_played_development_card:
            played_card_obj = self.game_manager.play_development_card(player, commerce_response)
            commerce_phase_object['trade_offer'] = 'played_card'
            commerce_phase_object['harbor_trade'] = False
            commerce_phase_object['development_card_played'] = played_card_obj
            if not (played_card_obj['played_card'] == 'victory_point' or
                    played_card_obj['played_card'] == 'failed_victory_point'):
                self.already_played_development_card = True

            return commerce_phase_object
        else:
            commerce_phase_object['trade_offer'] = 'None'
            return commerce_phase_object
        # TODO: fin de mover lógica a self.game_manager.on_commerce_response()
        # TODO: hacer un return de "commerce_phase_object" tras añadirle los datos de self.game_manager.on_commerce_response()

    def start_build_phase(self, player=-1):
        """
        Esta función permite pasar a la fase de construcción a un jugador.
        :param player: (int) número que representa al jugador.
        :return: void
        """

        build_phase_object = {}

        self.game_manager.set_phase(2)

        build_response = self.game_manager.call_to_bot_on_build_phase(player)

        # TODO: mover lógica a self.game_manager.on_build_response() -> devuelve un objeto similar a "build_phase_object"
        if isinstance(build_response, dict):
            build_phase_object = build_response
            built = False
            if build_response['building'] == BuildConstants.TOWN:
                built = self.game_manager.build_town(player, build_response['node_id'])
                if built['response']:
                    self.game_manager.get_players()[player]['victory_points'] += 1
            elif build_response['building'] == BuildConstants.CITY:
                built = self.game_manager.build_city(player, build_response['node_id'])
                if built['response']:
                    self.game_manager.bget_players()[player]['victory_points'] += 1
            elif build_response['building'] == BuildConstants.ROAD:
                built = self.game_manager.build_road(player, build_response['node_id'], build_response['road_to'])
            elif build_response['building'] == BuildConstants.CARD:
                built = self.game_manager.build_development_card(player)

            if isinstance(built, dict):
                if built['response']:
                    build_phase_object['finished'] = True

                    if build_response['building'] == BuildConstants.CARD:
                        build_phase_object['card_id'] = built['card_effect']
                        build_phase_object['card_type'] = built['card_type']
                        build_phase_object['card_effect'] = built['card_effect']

                    # Si se ha construido permitir que vuelvan a construir
                else:
                    build_phase_object['finished'] = False

                    # TODO: Avisar que no se ha podido construir
            else:
                build_phase_object['finished'] = False
                build_phase_object['error_msg'] = 'Falta de materiales'
                # TODO: Avisar que no se ha podido construir

        elif isinstance(build_response, DevelopmentCard) and not self.already_played_development_card:
            played_card_obj = self.game_manager.play_development_card(player, build_response)
            build_phase_object['building'] = 'played_card'
            build_phase_object['finished'] = 'played_card'
            build_phase_object['development_card_played'] = played_card_obj

            if not (played_card_obj['played_card'] == 'victory_point' or
                    played_card_obj['played_card'] == 'failed_victory_point'):
                self.already_played_development_card = True

            return build_phase_object
        else:
            build_phase_object['building'] = 'None'
        # TODO: fin de mover lógica a self.game_manager.on_build_response() -> devuelve un objeto similar a "build_phase_object"

        return build_phase_object

    def end_turn(self, player=-1):
        """
        Esta función permite finalizar el turno
        TODO: falta por comprobar si la partida se ha acabado aquí y no en el final de la ronda, para que solo haya un ganador
        :param player: número que representa al jugador
        :return: void
        """

        end_turn_object = {'development_card_played': []}

        self.game_manager.set_phase(3)

        turn_end_response = self.game_manager.call_to_bot_on_turn_end(player)

        if isinstance(turn_end_response, DevelopmentCard) and not self.already_played_development_card:
            played_card_obj = self.game_manager.play_development_card(player, turn_end_response)
            if not (played_card_obj['played_card'] == 'victory_point' or
                    played_card_obj['played_card'] == 'failed_victory_point'):
                self.already_played_development_card = True
            end_turn_object['development_card_played'].append(played_card_obj)

        # -- -- -- -- Calcular carretera más larga -- -- -- --
        # Le quitamos el título al jugador que lo tiene
        for player in self.game_manager.get_players():
            if player['longest_road'] == 1:
                player['longest_road'] = 0
                player['victory_points'] -= 2
                break

        # Calculamos quien tiene la carretera más larga
        real_longest_road = {'longest_road': 5, 'player': -1}
        for node in self.game_manager.get_board_nodes():
            longest_road_obj = self.game_manager.longest_road_calculator(node, 1, {'longest_road': 0, 'player': -1}, -1,
                                                                         [node['id']])

            if longest_road_obj['longest_road'] > real_longest_road['longest_road']:
                real_longest_road = longest_road_obj
        # Se le da el título a quien tenga la carretera más larga
        if real_longest_road['player'] != -1:
            self.game_manager.get_players()[real_longest_road['player']]['longest_road'] = 1
            self.game_manager.get_players()[real_longest_road['player']]['victory_points'] += 2

        vp = {}
        for i in range(4):
            vp['J' + str(i)] = str(self.game_manager.get_players()[i]['victory_points'])

        end_turn_object['victory_points'] = vp
        return end_turn_object

    # Round #
    def round_start(self):
        """
        Esta función permite comenzar una ronda nueva.
        """
        round_object = {}
        self.already_played_development_card = False

        for i in range(4):
            obj = {}
            self.game_manager.set_turn(self.game_manager.get_turn() + 1)
            self.game_manager.set_whose_turn_is_it(i)

            start_turn_object = self.start_turn(self.game_manager.get_whose_turn_is_it())
            obj['start_turn'] = start_turn_object

            # Se permite comerciar un máximo de 2 veces con jugadores, pero cualquier cantidad con el puerto.
            # Si se intenta comercia con un jugador una tercera vez, devuelve None y corta el bucle
            commerce_phase_array, depth = [], 1
            while True:
                commerce_phase_object = self.start_commerce_phase(self.game_manager.get_whose_turn_is_it(),
                                                                  depth)
                commerce_phase_array.append(commerce_phase_object)
                if commerce_phase_object['trade_offer'] == 'None':
                    break
                elif not (commerce_phase_object['harbor_trade'] or commerce_phase_object['harbor_trade'] is None):
                    depth += 1
            obj['commerce_phase'] = commerce_phase_array

            # Se puede construir cualquier cantidad de veces en un turno mientras tengan materiales. Así que
            # para evitar un bucle infinito, se corta si se construye 'None' o si fallan al intentar construir
            build_phase_array = []
            while True:
                build_phase_object = self.start_build_phase(self.game_manager.get_whose_turn_is_it())
                build_phase_array.append(build_phase_object)
                if build_phase_object['building'] == 'None' or not build_phase_object['finished']:
                    break
            obj['build_phase'] = build_phase_array

            end_turn_object = self.end_turn(self.game_manager.get_whose_turn_is_it())
            obj['end_turn'] = end_turn_object

            round_object['turn_P' + str(i)] = obj

        return round_object

    def round_end(self):
        """
        Esta función permite acabar una ronda empezada.
        """

        winner = False
        for player in self.game_manager.get_players():
            if player['victory_points'] >= 10:
                # if (player['victory_points'] >= 10 or
                # (player['victory_points'] >= 8 and (player['largest_army'] == 1 or player['longest_road'] == 1)) or
                # (player['victory_points'] >= 6 and player['largest_army'] == 1 and player['longest_road'] == 1)):
                winner = True

        self.game_manager.set_round(self.game_manager.get_round() + 1)
        return winner

    # Game #
    def game_start(self, game_number=0):
        """
        Esta función permite comenzar una partida nueva.
        :param game_number: (int) número de partidas que se van a jugar.
        """
        # Se cargan los bots y se inicializa el tablero
        # self.game_manager.bot_manager.load_bots()
        self.reset_game_values()

        # Se añade el tablero al setup, para que el intérprete sepa cómo es el tablero
        setup_object = {
            "board": {
                "board_nodes": self.game_manager.get_board_nodes(),
                "board_terrain": self.game_manager.get_board_terrain(),
            }
        }

        # Se le da paso al primer jugador para que ponga un poblado y una aldea
        for i in range(4):
            setup_object["P" + str(i)] = []

            self.game_manager.set_actual_player(i)
            self.game_manager.set_whose_turn_is_it(i)

            # función recursiva a introducir
            node_id, road_to = self.game_manager.on_game_start_build_towns_and_roads(i)
            setup_object["P" + str(i)].append({"id": node_id, "road": road_to})

        for i in range(3, -1, -1):
            self.game_manager.set_actual_player(i)
            self.game_manager.set_whose_turn_is_it(i)

            # función recursiva a introducir
            node_id, road_to = self.game_manager.on_game_start_build_towns_and_roads(i)
            setup_object["P" + str(i)].append({"id": node_id, "road": road_to})

        self.trace_loader.current_trace["setup"] = setup_object
        self.game_loop(game_number)
        return

    def game_loop(self, game_number):
        """
        Esta función permite jugar varias partidas seguidas.
        :param game_number: (int) número de partidas que se van a jugar.
        """
        game_object = {}
        winner = False
        while not winner:
            game_object['round_' + str(self.game_manager.get_round())] = self.round_start()
            winner = self.round_end()

        print('Game (' + str(game_number) + ') results')
        for i in range(4):
            print('J' + str(i) + ': ' + str(self.game_manager.get_players()[i]['victory_points']) + ' (' + str(
                self.game_manager.get_players()[i]['largest_army']) + ')' + ' (' + str(
                self.game_manager.get_players()[i]['longest_road']) + ')')

        self.trace_loader.current_trace["game"] = game_object
        self.trace_loader.export_to_file(game_number)
        return
