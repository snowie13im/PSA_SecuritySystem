import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    
    x = data["coord"]["x"]  # pixels relativos ao centro, positivo = direita
    y = data["coord"]["y"]
    print(f"Coordenadas recebidas: x={x}, y={y}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "colega-sub")
client.on_message = on_message
client.connect("localhost", 1883)  # mesmo endereço que usaste
client.subscribe("psahorus/facerec/resultado")
client.loop_forever()